# Converts crawled JSONL documents into a SQLite inverted index.
import json
import math
import sqlite3
import logging
import argparse
from collections import defaultdict, Counter
from pathlib import Path
from tokenizer import tokenize_document
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)
# BM25 constants control term-frequency saturation and document-length normalization.
BM25_K1 = 1.5
BM25_B = 0.75
INPUT_FILE = 'docs.jsonl'
DB_FILE = 'index.db'
SNIPPET_LEN = 200

# Store a short preview in SQLite so results can be displayed without full document text.
def extract_snippet(prose_text: str, max_len: int=SNIPPET_LEN) -> str:
    text = prose_text.strip()
    if len(text) <= max_len:
        return text
    cut = text[:max_len].rfind(' ')
    return text[:cut] + '...' if cut > 0 else text[:max_len] + '...'

# Rebuild the compact retrieval database from the JSONL corpus.
def create_tables(conn: sqlite3.Connection):
    conn.executescript('\n        DROP TABLE IF EXISTS documents;\n        DROP TABLE IF EXISTS postings;\n\n        CREATE TABLE documents (\n            doc_id      INTEGER PRIMARY KEY,\n            source      TEXT,\n            url         TEXT,\n            title       TEXT,\n            snippet     TEXT,\n            score       INTEGER,\n            is_accepted INTEGER\n        );\n\n        CREATE TABLE postings (\n            term    TEXT,\n            doc_id  INTEGER,\n            tf_idf  REAL,\n            bm25    REAL\n        );\n    ')
    log.info('Tables created.')

def create_indexes(conn: sqlite3.Connection):
    conn.executescript('\n        CREATE INDEX IF NOT EXISTS idx_postings_term   ON postings(term);\n        CREATE INDEX IF NOT EXISTS idx_postings_doc_id ON postings(doc_id);\n        CREATE INDEX IF NOT EXISTS idx_documents_id    ON documents(doc_id);\n    ')
    log.info('Indexes created.')

def build_index(input_file: str=INPUT_FILE, db_file: str=DB_FILE):
    log.info(f"Loading documents from '{input_file}' ...")
    docs = []
    with open(input_file, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    N = len(docs)
    log.info(f'Loaded {N} documents.')
    if N == 0:
        log.error('No documents found. Run crawler.py first.')
        return
    log.info('Tokenizing documents ...')
    # Document frequency is counted once per term per document for IDF.
    doc_tokens = []
    doc_tf = []
    df = defaultdict(int)
    for i, doc in enumerate(docs):
        tokens = tokenize_document(prose_text=doc.get('prose_text', ''), code_text=doc.get('code_text', ''))
        tf = Counter(tokens)
        doc_tokens.append(tokens)
        doc_tf.append(tf)
        for term in tf:
            df[term] += 1
        if (i + 1) % 100 == 0:
            log.info(f'  Tokenized {i + 1}/{N} documents ...')
    log.info(f'Tokenization complete. Vocabulary size: {len(df)} unique terms.')
    doc_lengths = [len(tokens) for tokens in doc_tokens]
    avg_doc_len = sum(doc_lengths) / N
    log.info(f'Average document length: {avg_doc_len:.1f} tokens')
    log.info('Computing TF-IDF and BM25 scores ...')
    # Each posting stores both TF-IDF and BM25 so experiments can switch modes.
    postings_rows = []
    for doc_id, (doc, tf, doc_len) in enumerate(zip(docs, doc_tf, doc_lengths)):
        title_tokens = set(tokenize_document(doc.get('title', ''), ''))
        for term, raw_count in tf.items():
            tf_score = raw_count / doc_len if doc_len > 0 else 0
            idf_score = math.log(N / df[term]) if df[term] > 0 else 0
            tf_idf = tf_score * idf_score
            idf_bm25 = math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1)
            tf_norm = raw_count * (BM25_K1 + 1) / (raw_count + BM25_K1 * (1 - BM25_B + BM25_B * doc_len / avg_doc_len))
            bm25 = idf_bm25 * tf_norm
            # Title terms are usually stronger relevance signals for technical questions.
            if term in title_tokens:
                bm25 *= 2.5
            postings_rows.append((term, doc_id, tf_idf, bm25))
    log.info(f'Computed {len(postings_rows)} posting entries.')
    log.info(f"Writing to '{db_file}' ...")
    conn = sqlite3.connect(db_file)
    # Insert documents first, then postings in batches for faster indexing.
    create_tables(conn)
    doc_rows = [(doc['id'], doc['source'], doc['url'], doc['title'], extract_snippet(doc.get('prose_text', '')), doc.get('score', 0), int(doc.get('is_accepted', False))) for doc in docs]
    conn.executemany('INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)', doc_rows)
    log.info(f'  Inserted {len(doc_rows)} documents.')
    BATCH = 10000
    for i in range(0, len(postings_rows), BATCH):
        conn.executemany('INSERT INTO postings VALUES (?, ?, ?, ?)', postings_rows[i:i + BATCH])
    log.info(f'  Inserted {len(postings_rows)} postings.')
    conn.commit()
    create_indexes(conn)
    conn.commit()
    conn.close()
    log.info(f"\nIndex built successfully → '{db_file}'")
    log.info(f'  Documents : {N}')
    log.info(f'  Unique terms : {len(df)}')
    log.info(f'  Total postings : {len(postings_rows)}')

def lookup(term: str, db_file: str=DB_FILE) -> list[tuple]:
    conn = sqlite3.connect(db_file)
    cursor = conn.execute('SELECT doc_id, tf_idf, bm25 FROM postings WHERE term = ? ORDER BY bm25 DESC', (term,))
    results = cursor.fetchall()
    conn.close()
    return results

def get_doc(doc_id: int, db_file: str=DB_FILE) -> dict:
    conn = sqlite3.connect(db_file)
    cursor = conn.execute('SELECT doc_id, source, url, title, snippet, score, is_accepted FROM documents WHERE doc_id = ?', (doc_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return {}
    return {'id': row[0], 'source': row[1], 'url': row[2], 'title': row[3], 'snippet': row[4], 'score': row[5], 'is_accepted': bool(row[6])}

def smoke_test(db_file: str=DB_FILE):
    print('\n' + '=' * 60)
    print('SMOKE TEST')
    print('=' * 60)
    test_queries = ['python', 'list', 'function', 'error', 'async']
    for term in test_queries:
        results = lookup(term, db_file)
        print(f"\n  '{term}' → {len(results)} documents found")
        for doc_id, tf_idf, bm25 in results[:3]:
            doc = get_doc(doc_id, db_file)
            print(f'    [{doc_id}] {doc.get('title', '?')[:50]:<50} | bm25={bm25:.3f}')
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='IIR indexer — CSC 575')
    ap.add_argument('--input', type=str, default=INPUT_FILE, help='Input .jsonl file')
    ap.add_argument('--db', type=str, default=DB_FILE, help='Output SQLite DB file')
    ap.add_argument('--test', action='store_true', help='Run smoke test after building')
    args = ap.parse_args()
    build_index(input_file=args.input, db_file=args.db)
    if args.test:
        smoke_test(args.db)