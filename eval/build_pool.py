# Builds an annotation pool by combining BM25, TF-IDF, boosted ranking, and Rocchio results.
import json
import sys
import argparse
import random
from pathlib import Path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from retrieval.engine import retrieve, retrieve_from_vector, _DEFAULT_DB
from retrieval.ranker import rerank
from indexer.tokenizer import tokenize_query
from interface.rocchio import rocchio_update
# Default output files used by the annotation workflow.
_DEFAULT_GT = str(Path(_PROJECT_ROOT) / 'data' / 'ground_truth.json')
_DEFAULT_POOL = str(Path(_PROJECT_ROOT) / 'data' / 'annotation_pool.json')
DEFAULT_QUERIES = ['python list comprehension', 'python data structures', 'python file read write', 'python error handling exceptions', 'python classes and objects', 'python functions arguments', 'python string methods', 'python dictionary operations', 'python lambda map filter', 'python generators iterators', 'python decorators', 'python modules import packages', 'python sorting algorithms', 'python regex regular expressions', 'python json parsing', 'async await javascript promise', 'javascript dom manipulation', 'javascript fetch api', 'TypeError list index out of range', 'AttributeError NoneType object']
POOL_DEPTH = 10

# Build comparable retrieval variants so the annotation pool is not biased by one method.
def _build_system_variants(db_file: str):

    def _run(q, mode='bm25', boost=True):
        raw = retrieve(q, mode=mode, top_k=POOL_DEPTH, db_file=db_file, boost=False)
        results = rerank(raw, query_tokens=tokenize_query(q)) if boost else raw
        return [{'doc_id': r.doc_id, 'title': r.title, 'url': r.url, 'snippet': r.snippet} for r in results]

    def _run_rocchio(q):
        tokens = tokenize_query(q)
        vector = {t: 1.0 for t in tokens}
        baseline = retrieve_from_vector(vector, mode='bm25', top_k=POOL_DEPTH, db_file=db_file, boost=False)
        # Simulate positive feedback using the first baseline results for Rocchio pooling.
        rel_ids = [r.doc_id for r in baseline[:3]]
        if not rel_ids:
            return [{'doc_id': r.doc_id, 'title': r.title, 'url': r.url, 'snippet': r.snippet} for r in baseline]
        updated = rocchio_update(vector, rel_ids, [], db_file=db_file)
        results = retrieve_from_vector(updated, mode='bm25', top_k=POOL_DEPTH, db_file=db_file, boost=False)
        results = rerank(results, query_tokens=tokens)
        return [{'doc_id': r.doc_id, 'title': r.title, 'url': r.url, 'snippet': r.snippet} for r in results]
    return {'bm25_boosted': lambda q: _run(q, mode='bm25', boost=True), 'bm25_baseline': lambda q: _run(q, mode='bm25', boost=False), 'tfidf': lambda q: _run(q, mode='tfidf', boost=False), 'rocchio': _run_rocchio}

def build_pool(queries: list, db_file: str, pool_file: str) -> dict:
    # Resume from an existing pool so manual labels are not lost.
    existing = {}
    try:
        with open(pool_file) as f:
            existing = json.load(f)
        print(f'Loaded existing pool with {len(existing)} queries.')
    except FileNotFoundError:
        pass
    variants = _build_system_variants(db_file)
    pool = dict(existing)
    for i, query in enumerate(queries, 1):
        if query in pool:
            print(f'[{i}/{len(queries)}] Skipping (already pooled): "{query}"')
            continue
        print(f'[{i}/{len(queries)}] Pooling: "{query}" ...', end=' ')
        seen_ids = set()
        merged = []
        # Merge unique documents from all systems before the user labels them.
        for sys_name, fn in variants.items():
            try:
                results = fn(query)
                for doc in results:
                    if doc['doc_id'] not in seen_ids:
                        seen_ids.add(doc['doc_id'])
                        merged.append({'doc_id': doc['doc_id'], 'title': doc['title'], 'url': doc['url'], 'snippet': doc['snippet'], 'relevant': None})
            except Exception as e:
                print(f'\n  Warning: {sys_name} failed for "{query}": {e}')
        random.shuffle(merged)
        pool[query] = merged
        print(f'{len(merged)} unique docs in pool')
    with open(pool_file, 'w') as f:
        json.dump(pool, f, indent=2)
    print(f'\nPool saved to {pool_file}')
    return pool

# Interactive labeling step used to produce ground_truth.json for evaluation.
def annotate(pool_file: str, gt_file: str):
    with open(pool_file) as f:
        pool = json.load(f)
    print('\n' + '=' * 65)
    print('  POOL-BASED ANNOTATION')
    print('  Commands: y=relevant  n=not relevant  s=skip  q=quit & save')
    print('=' * 65)
    total_queries = len(pool)
    completed = 0
    for q_idx, (query, docs) in enumerate(pool.items(), 1):
        unlabeled = [d for d in docs if d['relevant'] is None]
        if not unlabeled:
            completed += 1
            continue
        print(f'\n{'─' * 65}')
        print(f'Query {q_idx}/{total_queries}: "{query}"')
        print(f'  {len(unlabeled)} docs to label, {len(docs) - len(unlabeled)} already done')
        print(f'{'─' * 65}')
        quit_flag = False
        for doc in docs:
            if doc['relevant'] is not None:
                continue
            print(f'\n  Title   : {doc['title']}')
            print(f'  URL     : {doc['url']}')
            snippet = doc.get('snippet', '')[:200]
            print(f'  Snippet : {snippet}...')
            print()
            while True:
                ans = input('  Relevant? [y/n/s/q] → ').strip().lower()
                if ans == 'y':
                    doc['relevant'] = True
                    break
                elif ans == 'n':
                    doc['relevant'] = False
                    break
                elif ans == 's':
                    break
                elif ans == 'q':
                    quit_flag = True
                    break
                else:
                    print('  Please enter y, n, s, or q')
            if quit_flag:
                break
        with open(pool_file, 'w') as f:
            json.dump(pool, f, indent=2)
        if quit_flag:
            print('\nProgress saved. Run again with --resume to continue.')
            break
        all_labeled = all((d['relevant'] is not None for d in docs))
        if all_labeled:
            completed += 1
    _write_ground_truth(pool, gt_file)
    print(f'\nAnnotation progress: {completed}/{total_queries} queries fully labeled')
    print(f'Ground truth saved to {gt_file}')

def _write_ground_truth(pool: dict, gt_file: str):
    # Keep only documents marked relevant; skipped and non-relevant docs are excluded.
    gt = []
    for query, docs in pool.items():
        relevant_ids = [d['doc_id'] for d in docs if d['relevant'] is True]
        if relevant_ids:
            gt.append({'query': query, 'relevant_doc_ids': relevant_ids})
    with open(gt_file, 'w') as f:
        json.dump(gt, f, indent=2)
    print(f'  {len(gt)} queries with at least 1 relevant doc written.')
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Pool-based annotation — CSC 575')
    ap.add_argument('--queries', type=str, default=None, help='Text file with one query per line (default: built-in 20 queries)')
    ap.add_argument('--db', type=str, default=_DEFAULT_DB)
    ap.add_argument('--pool', type=str, default=_DEFAULT_POOL)
    ap.add_argument('--gt', type=str, default=_DEFAULT_GT)
    ap.add_argument('--pool-only', action='store_true', help='Build pool only, skip annotation')
    ap.add_argument('--label-only', action='store_true', help='Skip pool building, go straight to annotation')
    ap.add_argument('--resume', action='store_true', help='Resume interrupted annotation session')
    args = ap.parse_args()
    queries = DEFAULT_QUERIES
    if args.queries:
        with open(args.queries) as f:
            queries = [l.strip() for l in f if l.strip()]
    if not args.label_only:
        build_pool(queries=queries, db_file=args.db, pool_file=args.pool)
    if not args.pool_only:
        annotate(pool_file=args.pool, gt_file=args.gt)