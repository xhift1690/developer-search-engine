# Tokenization utilities shared by indexing, retrieval, and evaluation.
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
_stemmer = PorterStemmer()
_stopwords = set(stopwords.words('english'))
# Keep programming words even if they appear in the normal English stopword list.
PROGRAMMING_KEYWORDS = {'def', 'return', 'class', 'import', 'from', 'if', 'else', 'elif', 'for', 'while', 'in', 'not', 'and', 'or', 'is', 'try', 'except', 'with', 'as', 'pass', 'break', 'continue', 'lambda', 'yield', 'global', 'nonlocal', 'del', 'raise', 'assert', 'finally', 'true', 'false', 'none', 'function', 'const', 'let', 'var', 'async', 'await', 'this', 'new', 'typeof', 'instanceof', 'return', 'class', 'extends', 'import', 'export', 'default', 'null', 'undefined', 'then', 'catch', 'switch', 'case', 'api', 'url', 'http', 'get', 'set', 'list', 'dict', 'map', 'key', 'value', 'index', 'error', 'type', 'data', 'id', 'node', 'item'}
_stopwords = _stopwords - PROGRAMMING_KEYWORDS

def _dedup(tokens: list[str]) -> list[str]:
    seen, result = (set(), [])
    for t in tokens:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result

# Split identifiers like getUserById and user_id into searchable subterms.
def _split_identifier(token: str) -> list[str]:
    spaced = re.sub('(?<=[a-z])(?=[A-Z])', ' ', token)
    spaced = re.sub('(?<=[A-Z])(?=[A-Z][a-z])', ' ', spaced)
    parts = re.split('[\\s_.\\-]+', spaced)
    return [p.lower() for p in parts if len(p) > 1]

# Prose is lowercased, stopword-filtered, stemmed, and deduplicated.
def tokenize_prose(text: str) -> list[str]:
    if not text:
        return []
    text = text.lower()
    tokens = re.split('[^a-z0-9]+', text)
    result = []
    for token in tokens:
        if len(token) < 2:
            continue
        if token in _stopwords:
            continue
        result.append(_stemmer.stem(token))
    return _dedup(result)

# Code keeps original identifiers and also adds split identifier parts.
def tokenize_code(text: str) -> list[str]:
    if not text:
        return []
    raw_tokens = re.split('[\\s\\(\\)\\[\\]\\{\\},:;=<>!+\\-*/&|^~%\\"\'`\\\\@#]+', text)
    tokens = []
    for raw in raw_tokens:
        if not raw or len(raw) < 2:
            continue
        original = raw.lower()
        tokens.append(original)
        parts = _split_identifier(raw)
        for part in parts:
            if part != original:
                tokens.append(part)
    return _dedup(tokens)

# Merge prose and code tokens into one document vector for indexing.
def tokenize_document(prose_text: str, code_text: str) -> list[str]:
    prose_tokens = tokenize_prose(prose_text)
    code_tokens = tokenize_code(code_text)
    return _dedup(prose_tokens + code_tokens)

def tokenize_query(query: str) -> list[str]:
    prose_tokens = tokenize_prose(query)
    code_tokens = tokenize_code(query)
    return _dedup(prose_tokens + code_tokens)
if __name__ == '__main__':
    print('=' * 60)
    print('PROSE TOKENIZER')
    print('=' * 60)
    prose_examples = ['Lists are mutable sequences in Python', 'The function returns an error if the value is None', 'JavaScript async functions return a Promise', 'Use a for loop to iterate over items in a list']
    for text in prose_examples:
        print(f'\n  Input : {text}')
        print(f'  Output: {tokenize_prose(text)}')
    print('\n' + '=' * 60)
    print('CODE TOKENIZER')
    print('=' * 60)
    code_examples = ['getUserById(user_id)', 'def calculate_total_price(items):', 'const fetchUserData = async () => {}', 'TypeError: list index out of range', "addEventListener('click', handleButtonClick)", 'HTMLParser.feed(data)']
    for text in code_examples:
        print(f'\n  Input : {text}')
        print(f'  Output: {tokenize_code(text)}')
    print('\n' + '=' * 60)
    print('QUERY TOKENIZER  (what Bhuvanesh calls)')
    print('=' * 60)
    queries = ['get user by id', 'getUserById', 'python list comprehension', 'async await javascript', 'TypeError list index out of range']
    for q in queries:
        print(f'\n  Query : {q}')
        print(f'  Tokens: {tokenize_query(q)}')
    print('\n' + '=' * 60)
    print('FULL DOCUMENT  (prose + code merged)')
    print('=' * 60)
    prose = 'List comprehensions provide a concise way to create lists in Python'
    code = 'squares = [x**2 for x in range(10)]\ngetUserData(user_id)'
    print(f'\n  Prose : {prose}')
    print(f'  Code  : {code}')
    print(f'  Tokens: {tokenize_document(prose, code)}')