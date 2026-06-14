# Ingests Stack Overflow CSV data and normalizes it into the project JSONL format.
import json
import re
import argparse
import logging
from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)
_HERE = Path(__file__).resolve().parent.parent
# Default Stack Overflow CSV locations from the Kaggle dump.
DEFAULT_Q_FILE = str(_HERE / 'data' / 'so_raw' / 'Questions.csv')
DEFAULT_A_FILE = str(_HERE / 'data' / 'so_raw' / 'Answers.csv')
DEFAULT_OUTPUT = str(_HERE / 'docs.jsonl')
DEFAULT_MAX_DOCS = 5000
SNIPPET_LEN = 200

# Split Stack Overflow HTML into prose and code so both can be searched properly.
def _clean_html(html: str) -> tuple[str, str]:
    if not html or not isinstance(html, str):
        return ('', '')
    soup = BeautifulSoup(html, 'html.parser')
    code_parts = []
    for tag in soup.find_all(['pre', 'code']):
        text = tag.get_text()
        if text.strip():
            code_parts.append(text.strip())
        tag.decompose()
    prose = soup.get_text(' ', strip=True)
    prose = re.sub('\\s+', ' ', prose).strip()
    return (prose, '\n'.join(code_parts))

def _snippet(prose: str) -> str:
    text = prose.strip()
    if len(text) <= SNIPPET_LEN:
        return text
    cut = text[:SNIPPET_LEN].rfind(' ')
    return text[:cut] + '...' if cut > 0 else text[:SNIPPET_LEN] + '...'

# Append Stack Overflow questions and answers into the same docs.jsonl corpus.
def ingest(questions_file: str=DEFAULT_Q_FILE, answers_file: str=DEFAULT_A_FILE, output_file: str=DEFAULT_OUTPUT, max_docs: int=DEFAULT_MAX_DOCS, append: bool=True) -> int:
    log.info(f'Loading Questions from {questions_file} ...')
    try:
        q_df = pd.read_csv(questions_file, usecols=['Id', 'Score', 'Title', 'Body', 'AcceptedAnswerId'], encoding='latin-1', on_bad_lines='skip')
    except ValueError:
        q_df = pd.read_csv(questions_file, usecols=['Id', 'Score', 'Title', 'Body'], encoding='latin-1', on_bad_lines='skip')
        q_df['AcceptedAnswerId'] = None
    log.info(f'  Loaded {len(q_df):,} questions')
    # Accepted answer IDs are used later as metadata boosts in ranking.
    accepted_ids = set(q_df['AcceptedAnswerId'].dropna().astype(int).tolist())
    title_lookup = dict(zip(q_df['Id'], q_df['Title'].fillna('')))
    log.info(f'Loading Answers from {answers_file} ...')
    a_df = pd.read_csv(answers_file, usecols=['Id', 'ParentId', 'Score', 'Body'], encoding='latin-1', on_bad_lines='skip')
    log.info(f'  Loaded {len(a_df):,} answers')
    mode = 'a' if append else 'w'
    start_id = 0
    if append:
        # Continue document IDs after existing crawled/docs records.
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                start_id = sum((1 for _ in f))
        except FileNotFoundError:
            start_id = 0
    log.info(f'Writing to {output_file} (start_id={start_id}, max={max_docs}) ...')
    written = 0
    doc_id = start_id
    with open(output_file, mode, encoding='utf-8') as fh:
        for _, row in q_df.iterrows():
            if written >= max_docs:
                break
            prose, code = _clean_html(str(row.get('Body', '')))
            title = str(row.get('Title', '')).strip()
            if not title or not prose:
                continue
            doc = {'id': doc_id, 'source': 'stackoverflow', 'url': f'https://stackoverflow.com/questions/{int(row['Id'])}', 'title': title, 'prose_text': prose, 'code_text': code, 'score': int(row.get('Score', 0)), 'is_accepted': False, 'snippet': _snippet(prose)}
            fh.write(json.dumps(doc, ensure_ascii=False) + '\n')
            doc_id += 1
            written += 1
        log.info(f'  Wrote {written} questions')
        ans_written = 0
        for _, row in a_df.iterrows():
            if written >= max_docs:
                break
            prose, code = _clean_html(str(row.get('Body', '')))
            if not prose:
                continue
            answer_id = int(row['Id'])
            parent_id = int(row['ParentId'])
            is_accepted = answer_id in accepted_ids
            q_title = title_lookup.get(parent_id, '')
            title = f'Answer: {q_title}' if q_title else f'Answer to question {parent_id}'
            doc = {'id': doc_id, 'source': 'stackoverflow', 'url': f'https://stackoverflow.com/questions/{parent_id}#{answer_id}', 'title': title, 'prose_text': prose, 'code_text': code, 'score': int(row.get('Score', 0)), 'is_accepted': is_accepted, 'snippet': _snippet(prose)}
            fh.write(json.dumps(doc, ensure_ascii=False) + '\n')
            doc_id += 1
            written += 1
            ans_written += 1
        log.info(f'  Wrote {ans_written} answers ({sum((1 for _ in accepted_ids if _ in set(a_df['Id'])))} accepted)')
    log.info(f'Done. Total SO docs written: {written}')
    return written
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Stack Overflow CSV ingestion — CSC 575')
    ap.add_argument('--questions', type=str, default=DEFAULT_Q_FILE)
    ap.add_argument('--answers', type=str, default=DEFAULT_A_FILE)
    ap.add_argument('--output', type=str, default=DEFAULT_OUTPUT)
    ap.add_argument('--max', type=int, default=DEFAULT_MAX_DOCS, help='Max docs to write (default 5000). Use 0 for no limit.')
    ap.add_argument('--overwrite', action='store_true', help='Overwrite docs.jsonl instead of appending')
    args = ap.parse_args()
    ingest(questions_file=args.questions, answers_file=args.answers, output_file=args.output, max_docs=args.max if args.max > 0 else 10000000, append=not args.overwrite)