# Command-line search interface with optional relevance feedback.
import re
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass, field
# Make project imports work when this interface is launched directly.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from retrieval.engine import retrieve, retrieve_from_vector, Result, _DEFAULT_DB
from retrieval.ranker import rerank
from indexer.tokenizer import tokenize_query
from interface.rocchio import rocchio_update
# Use Rich for readable terminal output, but fall back to plain text if unavailable.
try:
    from rich.console import Console
    from rich.text import Text
    from rich.rule import Rule
    _console = Console()

    def _rule(title: str=''):
        _console.print(Rule(title))

    def _print(*args, **kwargs):
        _console.print(*args, **kwargs)

    def _input(prompt: str) -> str:
        return _console.input(prompt)

    def _highlight_terms(text: str, terms: list[str]):
        rich_text = Text(text)
        for term in terms:
            for m in re.finditer(re.escape(term), text, re.IGNORECASE):
                rich_text.stylize('bold magenta', m.start(), m.end())
        return rich_text
    _RICH = True
except ImportError:
    _RICH = False

    def _rule(title: str=''):
        print(f'\n{'─' * 60}' + (f'  {title}' if title else ''))

    def _print(*args, **kwargs):
        text = ' '.join((str(a) for a in args))
        text = re.sub('\\[/?[^\\]]+\\]', '', text)
        print(text)

    def _input(prompt: str) -> str:
        return input(re.sub('\\[/?[^\\]]+\\]', '', prompt))

    def _highlight_terms(text: str, terms: list[str]) -> str:
        return text

@dataclass
class QueryState:
    original_query: str
    terms: list = field(default_factory=list)
    original_vector: dict = field(default_factory=dict)
    current_vector: dict = field(default_factory=dict)
    relevant_ids: set = field(default_factory=set)
    nonrelevant_ids: set = field(default_factory=set)

# Select the snippet window with the densest query-term matches.
def extract_snippet(prose_text: str, query_terms: list[str], window: int=150) -> str:
    text = re.sub('<[^>]+>', '', prose_text)
    text = re.sub('&\\w+;', ' ', text)
    text = re.sub('\\s+', ' ', text).strip()
    if not query_terms or len(text) <= window:
        return text[:window]
    pattern = '|'.join((re.escape(t) for t in query_terms if t))
    if not pattern:
        return text[:window]
    positions = [m.start() for m in re.finditer(pattern, text, re.IGNORECASE)]
    if not positions:
        return text[:window]
    half = window // 2
    best_start, best_count = (0, 0)
    for pos in positions:
        start = max(0, pos - half)
        count = sum((1 for p in positions if start <= p < start + window))
        if count > best_count:
            best_count = count
            best_start = start
    return text[best_start:best_start + window]

# Show results and collect relevance judgments for Rocchio feedback.
def display_results(results: list[Result], query_terms: list[str], db_file: str=_DEFAULT_DB) -> dict[int, str]:
    judgments: dict[int, str] = {}
    grouped = sorted(results, key=lambda r: (0 if r.source in ('python_docs', 'mdn') else 1, -r.final_score))
    for rank, r in enumerate(grouped, 1):
        if r.source == 'python_docs':
            badge = '[bold cyan][PYDO][/]' if _RICH else '[PYDO]'
        elif r.source == 'mdn':
            badge = '[bold cyan][MDN ][/]' if _RICH else '[MDN ]'
        else:
            badge = '[bold yellow][SO  ][/]' if _RICH else '[SO  ]'
        accepted_mark = (' [green]✓[/]' if _RICH else ' ✓') if r.is_accepted else ''
        _rule()
        _print(f'[bold]#{rank}[/] {badge} [bold]{r.title}[/]{accepted_mark}')
        _print(f'[dim]{r.url}[/]')
        _print(f'Score: {r.final_score:.4f}  Raw: {r.raw_score:.4f}  Votes: {r.votes}')
        _print(_highlight_terms(extract_snippet(r.snippet or '', query_terms), query_terms))
        verdict = _input('\n  [r]elevant / [n]ot-relevant / [s]kip / [q]uit: ').strip().lower()
        if verdict == 'q':
            judgments[r.doc_id] = 'q'
            break
        judgments[r.doc_id] = verdict
    return judgments
# Limit feedback rounds so the demo stays short and repeatable.
MAX_ROUNDS = 2

def feedback_loop(query_state: QueryState, mode: str='bm25', top_k: int=10, db_file: str=_DEFAULT_DB, boost: bool=True):
    for round_num in range(MAX_ROUNDS):
        raw_results = retrieve_from_vector(query_state.current_vector, mode=mode, top_k=top_k, db_file=db_file, boost=False)
        results = rerank(raw_results, query_tokens=query_state.terms) if boost else raw_results
        header = '— Re-ranked —' if round_num > 0 else f'Results for: "{query_state.original_query}"'
        _rule(header)
        judgments = display_results(results, query_state.terms, db_file=db_file)
        for doc_id, verdict in judgments.items():
            if verdict == 'r':
                query_state.relevant_ids.add(doc_id)
            elif verdict == 'n':
                query_state.nonrelevant_ids.add(doc_id)
        if not query_state.relevant_ids and (not query_state.nonrelevant_ids):
            break
        # Update the query vector from the user's relevance judgments.
        query_state.current_vector = rocchio_update(query_state.original_vector, list(query_state.relevant_ids), list(query_state.nonrelevant_ids), db_file=db_file)
        if round_num + 1 == MAX_ROUNDS:
            final_raw = retrieve_from_vector(query_state.current_vector, mode=mode, top_k=top_k, db_file=db_file, boost=False)
            final = rerank(final_raw, query_tokens=query_state.terms) if boost else final_raw
            _rule('— Final Re-ranked Results —')
            for rank, r in enumerate(final, 1):
                acc = '✓' if r.is_accepted else ' '
                _print(f'  {rank:>2}. [{r.source:<12}] {acc} {r.title[:55]:<55} score={r.final_score:.4f}')

def search_session(query: str, mode: str='bm25', top_k: int=10, db_file: str=_DEFAULT_DB, boost: bool=True):
    tokens = tokenize_query(query)
    vector = {t: 1.0 for t in tokens}
    state = QueryState(original_query=query, terms=tokens, original_vector=dict(vector), current_vector=vector)
    feedback_loop(state, mode=mode, top_k=top_k, db_file=db_file, boost=boost)
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='IIR display — CSC 575')
    ap.add_argument('query', type=str)
    ap.add_argument('--mode', type=str, default='bm25', choices=['bm25', 'tfidf'])
    ap.add_argument('--top', type=int, default=10)
    ap.add_argument('--db', type=str, default=_DEFAULT_DB)
    ap.add_argument('--no-boost', action='store_true')
    args = ap.parse_args()
    search_session(args.query, mode=args.mode, top_k=args.top, db_file=args.db, boost=not args.no_boost)