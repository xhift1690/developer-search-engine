# Retrieval engine for BM25/TF-IDF search over the SQLite index.
import sqlite3
import heapq
import math
import sys
import argparse
from dataclasses import dataclass, field
from pathlib import Path
# Resolve tokenizer imports consistently from scripts and notebooks.
_INDEXER_DIR = Path(__file__).resolve().parent.parent / 'indexer'
if str(_INDEXER_DIR) not in sys.path:
    sys.path.insert(0, str(_INDEXER_DIR))
from tokenizer import tokenize_query
_DEFAULT_DB = str(Path(__file__).resolve().parent.parent / 'data' / 'index.db')
# Metadata boosts are optional and can be disabled for baseline experiments.
BOOST_ACCEPTED = 1.3
BOOST_DOCS = 1.2
BOOST_VOTE_COEF = 0.1

@dataclass
class Result:
    doc_id: int
    title: str
    url: str
    snippet: str
    final_score: float
    raw_score: float
    is_accepted: bool
    votes: int
    source: str

@dataclass
class QueryState:
    original_query: str
    current_vector: dict = field(default_factory=dict)
    relevant_ids: list = field(default_factory=list)
    nonrelevant_ids: list = field(default_factory=list)
# Reuse SQLite connections so repeated searches are faster.
_conn_pool: dict[str, sqlite3.Connection] = {}

def _get_conn(db_file: str) -> sqlite3.Connection:
    if db_file not in _conn_pool:
        conn = sqlite3.connect(db_file, check_same_thread=False)
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA cache_size=-65536')
        conn.execute('PRAGMA temp_store=MEMORY')
        _conn_pool[db_file] = conn
    return _conn_pool[db_file]

def build_query_vector(query_str: str) -> dict[str, float]:
    tokens = tokenize_query(query_str)
    return {t: 1.0 for t in tokens}

def retrieve(query_str: str, mode: str='bm25', top_k: int=10, db_file: str=_DEFAULT_DB, boost: bool=True) -> list[Result]:
    vector = build_query_vector(query_str)
    if not vector:
        return []
    return retrieve_from_vector(vector, mode=mode, top_k=top_k, db_file=db_file, boost=boost)

# Core retrieval path used by normal search and Rocchio-updated query vectors.
def retrieve_from_vector(query_vector: dict[str, float], mode: str='bm25', top_k: int=10, db_file: str=_DEFAULT_DB, boost: bool=True) -> list[Result]:
    if not query_vector:
        return []
    conn = _get_conn(db_file)
    score_col = 'bm25' if mode == 'bm25' else 'tf_idf'
    scores: dict[int, float] = {}
    for term, q_weight in query_vector.items():
        if q_weight <= 0:
            continue
        rows = conn.execute(f'SELECT doc_id, {score_col} FROM postings WHERE term = ?', (term,)).fetchall()
        for doc_id, doc_score in rows:
            scores[doc_id] = scores.get(doc_id, 0.0) + q_weight * doc_score
    if not scores:
        return []
    # Pull extra candidates before metadata boosting and final trimming.
    candidates = heapq.nlargest(top_k * 3, scores.items(), key=lambda x: x[1])
    candidate_ids = [doc_id for doc_id, _ in candidates]
    placeholders = ','.join('?' * len(candidate_ids))
    doc_rows = conn.execute(f'SELECT doc_id, title, url, snippet, score, is_accepted, source FROM documents WHERE doc_id IN ({placeholders})', candidate_ids).fetchall()
    docs_by_id = {row[0]: row for row in doc_rows}
    results = []
    for doc_id, raw_score in candidates:
        if doc_id not in docs_by_id:
            continue
        row = docs_by_id[doc_id]
        votes = int(row[4]) if row[4] else 0
        is_accepted = bool(row[5])
        source = row[6] or ''
        final = raw_score
        if boost:
            # Favor accepted answers, official docs, and high-vote Stack Overflow posts.
            if is_accepted:
                final *= BOOST_ACCEPTED
            if source in ('python_docs', 'mdn'):
                final *= BOOST_DOCS
            if votes > 0:
                final += math.log(1 + votes) * BOOST_VOTE_COEF
        results.append(Result(doc_id=doc_id, title=row[1] or '(no title)', url=row[2] or '', snippet=row[3] or '', final_score=final, raw_score=raw_score, is_accepted=is_accepted, votes=votes, source=source))
    results.sort(key=lambda r: r.final_score, reverse=True)
    return results[:top_k]

# Used by Rocchio to retrieve a document's term-weight vector.
def get_doc_vector(doc_id: int, db_file: str=_DEFAULT_DB) -> dict[str, float]:
    conn = _get_conn(db_file)
    rows = conn.execute('SELECT term, tf_idf FROM postings WHERE doc_id = ?', (doc_id,)).fetchall()
    return {term: tf_idf for term, tf_idf in rows}

def _smoke_test(db_file: str=_DEFAULT_DB):
    test_queries = ['python list comprehension', 'async await javascript', 'TypeError list index out of range', 'how to sort a dictionary by value', 'fetch API promise JavaScript']
    src_tag = {'stackoverflow': 'SO  ', 'python_docs': 'PYDO', 'mdn': 'MDN '}
    for mode in ('bm25', 'tfidf'):
        print(f'\n{'=' * 70}\n  MODE: {mode.upper()}\n{'=' * 70}')
        for q in test_queries:
            results = retrieve(q, mode=mode, top_k=5, db_file=db_file)
            print(f'\n  Query: "{q}"')
            for i, r in enumerate(results, 1):
                tag = src_tag.get(r.source, '????')
                check = '✓' if r.is_accepted else ' '
                print(f'  {i:<3} {tag}  {r.final_score:>7.3f}  {check} {r.title[:55]}')
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='IIR retrieval engine — CSC 575')
    ap.add_argument('--query', type=str, default=None)
    ap.add_argument('--mode', type=str, default='bm25', choices=['bm25', 'tfidf'])
    ap.add_argument('--top', type=int, default=10)
    ap.add_argument('--db', type=str, default=_DEFAULT_DB)
    ap.add_argument('--no-boost', action='store_true')
    args = ap.parse_args()
    if args.query:
        results = retrieve(args.query, mode=args.mode, top_k=args.top, db_file=args.db, boost=not args.no_boost)
        print(f'\nQuery : "{args.query}"')
        print(f'Mode  : {args.mode}  |  Boost: {not args.no_boost}  |  Top {args.top}')
        print('=' * 70)
        for i, r in enumerate(results, 1):
            check = '✓' if r.is_accepted else ' '
            print(f'  {i:>2}. [{r.source:<12}] {check} {r.title[:55]:<55} score={r.final_score:.4f}')
            print(f'      {r.url}')
    else:
        _smoke_test(args.db)