# Developer Search Engine — CSC 575 Information Retrieval

A search engine for Python and JavaScript developer questions. It indexes Stack Overflow answers, Python docs, and MDN Web Docs, then retrieves results using BM25 and TF-IDF with metadata re-ranking and Rocchio relevance feedback.

Built entirely with classical IR techniques — no neural models, no embeddings, just an inverted index in SQLite.

---

## How it works

```
Stack Overflow CSVs  +  Python Docs  +  MDN Web Docs
              │
              ▼
     so_ingest.py / crawler.py
     (clean HTML, separate prose from code)
              │
              ▼
     tokenizer.py
     (lowercase → camelCase split → stopword removal → Porter stemming)
              │
              ▼
     indexer_not.py
     (build SQLite inverted index with TF-IDF and BM25 scores)
              │
              ▼
     engine.py  →  ranker.py
     (TAAT retrieval + metadata boost: accepted answers, vote score, source type, title match)
              │
              ▼
     display.py  +  rocchio.py
     (interactive search with relevance feedback loop)
              │
              ▼
     build_pool.py  →  run_eval.py
     (pool-based annotation → P@5, P@10, MRR, NDCG@10)
```

---

## Project structure

```
Project/
    indexer/
        crawler.py         crawls Python docs and MDN, writes docs.jsonl
        so_ingest.py       converts Kaggle Stack Overflow CSVs to docs.jsonl
        tokenizer.py       shared tokenizer (used by indexer, retrieval, and eval)
        indexer_not.py     builds the SQLite inverted index from docs.jsonl

    retrieval/
        engine.py          BM25/TF-IDF TAAT retrieval from SQLite
        ranker.py          metadata re-ranking on top of raw retrieval scores

    interface/
        display.py         interactive CLI search with feedback loop
        rocchio.py         Rocchio query expansion

    eval/
        build_pool.py      builds annotation pool from multiple system variants
        run_eval.py        P@5, P@10, MRR, NDCG@10 evaluation

    data/
        annotation_pool.json
        ground_truth.json
        index.db

    docs.jsonl             cleaned document collection
    index.db               main search index
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate       # Mac/Linux
# .venv\Scripts\activate        # Windows

pip install pandas beautifulsoup4 requests nltk numpy scikit-learn rich
```

Uses only Python built-ins otherwise (`sqlite3`, `json`, `math`, `argparse`, `pathlib`).

---

## Dataset

Download the **Python Questions from Stack Overflow** dataset from Kaggle:  
https://www.kaggle.com/datasets/stackoverflow/pythonquestions

Place the files here before running ingestion:

```
Project/
    data/
        so_raw/
            Questions.csv
            Answers.csv
            Tags.csv
```

---

## Running it

All commands run from inside `Project/Project/`:

```bash
cd Project/Project
```

### 1 — Ingest Stack Overflow data

```bash
python indexer/so_ingest.py \
  --questions data/so_raw/Questions.csv \
  --answers data/so_raw/Answers.csv \
  --output docs.jsonl \
  --max 5000
```

### 2 — Crawl Python and MDN docs

```bash
python indexer/crawler.py
```

Respects `robots.txt` and adds a 1.5s delay between requests.

### 3 — Build the index

```bash
python indexer/indexer_not.py --input docs.jsonl --db data/index.db --test
```

Creates two SQLite tables: `documents` (metadata) and `postings` (term-level TF-IDF and BM25 scores).

### 4 — Search

```bash
python retrieval/engine.py --query "python list comprehension" --mode bm25 --top 10 --db data/index.db
python retrieval/engine.py --query "async await javascript" --mode tfidf --top 10 --db data/index.db
```

### 5 — Interactive search with relevance feedback

```bash
python interface/display.py "TypeError list index out of range" --mode bm25 --top 10 --db data/index.db
```

Mark results relevant or not, and Rocchio updates the query vector for the next round.

### 6 — Evaluate

```bash
python eval/build_pool.py --db data/index.db
python eval/run_eval.py --gt data/ground_truth.json --db data/index.db
```

Compares: TF-IDF vs BM25, no boost vs metadata boost, baseline vs post-Rocchio.

---

## Techniques used

| Technique | File |
|---|---|
| Code-aware tokenization (camelCase, snake_case splitting) | `tokenizer.py` |
| Stop-word removal with programming keyword exceptions | `tokenizer.py` |
| Porter stemming | `tokenizer.py` |
| Inverted index in SQLite | `indexer_not.py` |
| TF-IDF scoring | `indexer_not.py`, `engine.py` |
| BM25 scoring (k1=1.5, b=0.75) | `indexer_not.py`, `engine.py` |
| Term-at-a-time (TAAT) retrieval | `engine.py` |
| Metadata re-ranking (accepted answers, vote score, source boost, title match) | `ranker.py` |
| Rocchio relevance feedback (α=1.0, β=0.75, γ=0.15) | `rocchio.py` |
| Pool-based annotation and evaluation | `build_pool.py`, `run_eval.py` |

---

## Example queries

```
python list comprehension
python dictionary operations
python error handling exceptions
TypeError list index out of range
AttributeError NoneType object
async await javascript promise
javascript fetch api
javascript dom manipulation
```

---

## Limitations

Results depend on which documents got indexed. The `--max 5000` cap on Stack Overflow ingestion keeps things manageable but means not every question is covered. Stack Overflow posts mix prose, HTML, and code, so some noise gets through even after cleaning. The system is purely lexical — it won't handle semantic rephrasing the way a neural search system would.
