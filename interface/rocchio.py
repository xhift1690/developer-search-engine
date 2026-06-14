# Rocchio relevance-feedback update for expanding/refining a query vector.
import sys
from pathlib import Path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from retrieval.engine import get_doc_vector, _DEFAULT_DB
# Rocchio weights: original query, relevant centroid, and non-relevant centroid.
ALPHA = 1.0
BETA = 0.75
GAMMA = 0.15

def rocchio_update(q_orig: dict[str, float], relevant_ids: list[int], nonrelevant_ids: list[int], db_file: str=_DEFAULT_DB, alpha: float=ALPHA, beta: float=BETA, gamma: float=GAMMA) -> dict[str, float]:

    # Average document vectors to represent the relevant or non-relevant set.
    def centroid(doc_ids: list[int]) -> dict[str, float]:
        if not doc_ids:
            return {}
        vec: dict[str, float] = {}
        for doc_id in doc_ids:
            for term, weight in get_doc_vector(doc_id, db_file=db_file).items():
                vec[term] = vec.get(term, 0.0) + weight
        n = len(doc_ids)
        return {t: w / n for t, w in vec.items()}
    R_centroid = centroid(relevant_ids)
    NR_centroid = centroid(nonrelevant_ids)
    # Combine all candidate terms, then drop negative weights after the update.
    all_terms = set(q_orig) | set(R_centroid) | set(NR_centroid)
    q_new: dict[str, float] = {}
    for term in all_terms:
        val = alpha * q_orig.get(term, 0.0) + beta * R_centroid.get(term, 0.0) - gamma * NR_centroid.get(term, 0.0)
        if val > 0:
            q_new[term] = val
    return q_new