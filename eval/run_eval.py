# Evaluates retrieval quality using P@5, P@10, MRR, and NDCG@10.
import json
import sys
import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics import ndcg_score as _sklearn_ndcg
# Make project modules importable when this script is run from eval/.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from retrieval.engine import retrieve, retrieve_from_vector, Result, _DEFAULT_DB
from retrieval.ranker import rerank
from indexer.tokenizer import tokenize_query
from interface.rocchio import rocchio_update
_DEFAULT_GT = str(Path(_PROJECT_ROOT) / 'data' / 'ground_truth.json')

# Standard IR metrics used to compare retrieval experiments.
def precision_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    if k == 0:
        return 0.0
    return sum((1 for doc in retrieved[:k] if doc in relevant)) / k

def mrr(retrieved: list[int], relevant: set[int]) -> float:
    for rank, doc in enumerate(retrieved, 1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0

def ndcg_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    relevance = [1 if doc in relevant else 0 for doc in retrieved[:k]]
    while len(relevance) < k:
        relevance.append(0)
    ideal = sorted(relevance, reverse=True)
    if not any(ideal):
        return 0.0
    return float(_sklearn_ndcg([ideal], [relevance]))

def evaluate(ground_truth_path: str, retrieve_fn) -> dict:
    # retrieve_fn lets the same metrics run against different ranking settings.
    with open(ground_truth_path) as f:
        queries = json.load(f)
    p5_list, p10_list, mrr_list, ndcg_list = ([], [], [], [])
    for item in queries:
        results = retrieve_fn(item['query'])
        retrieved = [r.doc_id if isinstance(r, Result) else r['doc_id'] for r in results[:20]]
        relevant = set(item['relevant_doc_ids'])
        p5_list.append(precision_at_k(retrieved, relevant, 5))
        p10_list.append(precision_at_k(retrieved, relevant, 10))
        mrr_list.append(mrr(retrieved, relevant))
        ndcg_list.append(ndcg_at_k(retrieved, relevant, 10))
    return {'P@5': float(np.mean(p5_list)), 'P@10': float(np.mean(p10_list)), 'MRR': float(np.mean(mrr_list)), 'NDCG@10': float(np.mean(ndcg_list))}

def print_table(rows: list[tuple]):
    print(f'\n| {'Experiment':<35} | {'P@5':>6} | {'P@10':>6} | {'MRR':>6} | {'NDCG@10':>8} |')
    print('|' + '-' * 37 + '|' + '-' * 8 + '|' + '-' * 8 + '|' + '-' * 8 + '|' + '-' * 10 + '|')
    for name, m in rows:
        print(f'| {name:<35} | {m['P@5']:>6.3f} | {m['P@10']:>6.3f} | {m['MRR']:>6.3f} | {m['NDCG@10']:>8.3f} |')

# Runs the three project comparisons: model choice, metadata boost, and Rocchio feedback.
def run_all_experiments(gt_path: str=_DEFAULT_GT, db_file: str=_DEFAULT_DB):
    with open(gt_path) as f:
        gt_data = json.load(f)
    gt_lookup: dict[str, set[int]] = {item['query']: set(item['relevant_doc_ids']) for item in gt_data}

    def _retrieve(q, mode='bm25', boost=True) -> list[Result]:
        raw = retrieve(q, mode=mode, top_k=20, db_file=db_file, boost=False)
        return rerank(raw, query_tokens=tokenize_query(q)) if boost else raw

    def _retrieve_rocchio(q) -> list[Result]:
        tokens = tokenize_query(q)
        vector = {t: 1.0 for t in tokens}
        baseline = retrieve_from_vector(vector, mode='bm25', top_k=20, db_file=db_file, boost=False)
        # For evaluation only, use ground truth labels to simulate relevant feedback.
        gt_rel = gt_lookup.get(q, set())
        rel_ids = [r.doc_id for r in baseline if r.doc_id in gt_rel]
        if not rel_ids:
            return baseline
        updated_vector = rocchio_update(vector, rel_ids, [], db_file=db_file)
        updated_raw = retrieve_from_vector(updated_vector, mode='bm25', top_k=20, db_file=db_file, boost=False)
        return rerank(updated_raw, query_tokens=tokens)
    rows = []
    print('Running Experiment 1: TF-IDF vs BM25 ...')
    rows.append(('TF-IDF', evaluate(gt_path, lambda q: _retrieve(q, mode='tfidf'))))
    rows.append(('BM25', evaluate(gt_path, lambda q: _retrieve(q, mode='bm25'))))
    print('Running Experiment 2: Metadata boost off vs on ...')
    rows.append(('No metadata boost', evaluate(gt_path, lambda q: _retrieve(q, boost=False))))
    rows.append(('Metadata boost', evaluate(gt_path, lambda q: _retrieve(q, boost=True))))
    print('Running Experiment 3: Baseline vs Rocchio ...')
    rows.append(('Baseline (pre-Rocchio)', evaluate(gt_path, lambda q: _retrieve(q))))
    rows.append(('After Rocchio feedback', evaluate(gt_path, _retrieve_rocchio)))
    print_table(rows)
    return rows
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='IIR evaluation — CSC 575')
    ap.add_argument('--gt', type=str, default=_DEFAULT_GT, help='Path to ground_truth.json')
    ap.add_argument('--db', type=str, default=_DEFAULT_DB, help='Path to index.db')
    args = ap.parse_args()
    run_all_experiments(gt_path=args.gt, db_file=args.db)