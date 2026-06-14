# Extra reranking layer for metadata, vote, and title-match signals.
import math
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'indexer'))
from engine import retrieve, Result, _DEFAULT_DB
from tokenizer import tokenize_query
# Reranking weights mirror the retrieval engine and are toggled during evaluation.
BOOST_ACCEPTED = 1.3
BOOST_DOCS = 1.2
BOOST_VOTE_COEF = 0.1
BOOST_TITLE_MATCH = 0.15

@dataclass
class ScoreBreakdown:
    raw_score: float
    accepted_mult: float
    docs_mult: float
    vote_bonus: float
    title_bonus: float
    final_score: float
    title_hits: int
    votes: int
    is_accepted: bool
    source: str

# Apply metadata and title-match boosts after the raw retrieval stage.
def rerank(results: list[Result], query_tokens: list[str]=None, boost_flags: dict=None) -> list[Result]:
    if not results:
        return []
    flags = {'accepted': True, 'docs': True, 'votes': True, 'title': True}
    if boost_flags:
        flags.update(boost_flags)
    q_terms = set(query_tokens) if query_tokens else set()
    for r in results:
        score = r.raw_score
        # Multiplicative boosts preserve raw-score ordering better than fixed bonuses.
        accepted_mult = BOOST_ACCEPTED if flags['accepted'] and r.is_accepted else 1.0
        score *= accepted_mult
        docs_mult = BOOST_DOCS if flags['docs'] and r.source in ('python_docs', 'mdn') else 1.0
        score *= docs_mult
        vote_bonus = math.log(1 + r.votes) * BOOST_VOTE_COEF if flags['votes'] and r.votes > 0 else 0.0
        score += vote_bonus
        title_bonus = 0.0
        # Title matches help exact technical terms surface near the top.
        if flags['title'] and q_terms and r.title:
            title_hits = len(q_terms & set(_title_tokens(r.title.lower())))
            title_bonus = title_hits * BOOST_TITLE_MATCH
        score += title_bonus
        r.final_score = score
    results.sort(key=lambda r: r.final_score, reverse=True)
    return results

def _title_tokens(title: str) -> list[str]:
    import re
    return [t for t in re.split('[^a-z0-9]+', title) if len(t) > 1]

# Return the score components used by the demo/debug output.
def explain(result: Result, query_tokens: list[str]=None) -> ScoreBreakdown:
    q_terms = set(query_tokens) if query_tokens else set()
    title_hits = len(q_terms & set(_title_tokens(result.title.lower()))) if q_terms and result.title else 0
    accepted_mult = BOOST_ACCEPTED if result.is_accepted else 1.0
    docs_mult = BOOST_DOCS if result.source in ('python_docs', 'mdn') else 1.0
    vote_bonus = math.log(1 + result.votes) * BOOST_VOTE_COEF if result.votes > 0 else 0.0
    title_bonus = title_hits * BOOST_TITLE_MATCH
    final = result.raw_score * accepted_mult * docs_mult + vote_bonus + title_bonus
    return ScoreBreakdown(raw_score=result.raw_score, accepted_mult=accepted_mult, docs_mult=docs_mult, vote_bonus=vote_bonus, title_bonus=title_bonus, final_score=final, title_hits=title_hits, votes=result.votes, is_accepted=result.is_accepted, source=result.source)

def _smoke_test(query: str, db_file: str=_DEFAULT_DB):
    tokens = tokenize_query(query)
    raw = retrieve(query, mode='bm25', top_k=10, db_file=db_file, boost=False)
    ranked = rerank(raw, query_tokens=tokens)
    src_label = {'stackoverflow': 'SO  ', 'python_docs': 'PYDO', 'mdn': 'MDN '}
    print(f'\nQuery  : "{query}"\nTokens : {tokens}')
    print(f'\n{'Rank':<5} {'Src':<5} {'Score':>8}  {'Raw':>8}  {'Acc':>4} {'Votes':>6}  Title')
    print('─' * 75)
    for i, r in enumerate(ranked, 1):
        bd = explain(r, tokens)
        check = '✓' if r.is_accepted else ' '
        src = src_label.get(r.source, '?   ')
        print(f'  {i:<3} {src}  {r.final_score:>8.3f}  {r.raw_score:>8.3f}  {check}    {r.votes:>5}  {r.title[:45]}')
        print(f'       acc×{bd.accepted_mult}  docs×{bd.docs_mult}  votes+{bd.vote_bonus:.3f}  title+{bd.title_bonus:.3f}')
    print('\n── Baseline (no boosts) ──')
    no_boost = retrieve(query, mode='bm25', top_k=10, db_file=db_file, boost=False)
    rerank(no_boost, boost_flags={'accepted': False, 'docs': False, 'votes': False, 'title': False})
    for i, r in enumerate(no_boost, 1):
        print(f'  {i:<3} {src_label.get(r.source, '?   ')}  {r.final_score:>8.3f}  {r.title[:50]}')
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='IIR ranker — CSC 575')
    ap.add_argument('--query', type=str, default='python list comprehension')
    ap.add_argument('--db', type=str, default=_DEFAULT_DB)
    args = ap.parse_args()
    _smoke_test(args.query, args.db)