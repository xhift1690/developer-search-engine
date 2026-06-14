# Focused crawler for collecting Python and JavaScript documentation pages.
import json
import time
import re
import logging
import urllib.robotparser
from urllib.parse import urlparse, urljoin, urldefrag
from collections import deque
from dataclasses import dataclass, asdict
import requests
from bs4 import BeautifulSoup
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)
# Seed pages define the focused crawl scope for Python and JavaScript documentation.
START_URLS = ['https://docs.python.org/3/tutorial/index.html', 'https://docs.python.org/3/library/index.html', 'https://docs.python.org/3/reference/index.html', 'https://docs.python.org/3/howto/index.html', 'https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide', 'https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference', 'https://developer.mozilla.org/en-US/docs/Web/API', 'https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects', 'https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Statements', 'https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators', 'https://developer.mozilla.org/en-US/docs/Web/HTML/Element', 'https://developer.mozilla.org/en-US/docs/Web/CSS/Reference', 'https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API', 'https://developer.mozilla.org/en-US/docs/Web/API/Document_Object_Model', 'https://developer.mozilla.org/en-US/docs/Web/API/Window', 'https://developer.mozilla.org/en-US/docs/Web/API/EventTarget', 'https://developer.mozilla.org/en-US/docs/Web/API/Promise', 'https://developer.mozilla.org/en-US/docs/Learn/JavaScript']
ALLOWED_DOMAINS = {'docs.python.org', 'developer.mozilla.org'}
# Skip navigation, marketing, and obsolete pages that do not help technical retrieval.
MDN_SKIP_PREFIXES = ('/en-US/search', '/en-US/plus', '/en-US/curriculum', '/en-US/blog', '/en-US/observatory', '/en-US/docs/Mozilla', '/en-US/docs/Tools')
JUNK_TITLES = {'copyright', 'license', 'about this documentation', 'download python', 'python 2', "what's new", 'glossary', 'genindex', 'modindex', 'search', '404', 'page not found', '403', 'forbidden', 'python 3.5', 'python 3.6', 'python 3.7', 'python 3.8', 'python 2', 'frequently asked questions', 'python howtos', 'python faq', 'deprecations', 'whats new', "what's new", 'changelog', 'python module index', 'python documentation contents', 'python/c api', 'extending and embedding', 'dealing with bugs', 'networking and interprocess', 'concurrent execution', 'file and directory access', 'data persistence', 'data compression', 'internet protocols', 'structured markup', 'numeric and mathematical', 'functional programming', 'generic operating system', 'internet data handling', 'graphical user interfaces', 'text processing', 'binary data services', 'multimedia services', 'internationalization', 'cryptographic services', 'command-line interface', 'what now', 'appendix', 'interactive input', 'debugging and profiling', 'development tools', 'audit events', 'improve a documentation', 'xml processing'}
REQUEST_DELAY = 1.5
MAX_PAGES = 10000
REQUEST_TIMEOUT = 15
OUTPUT_FILE = 'docs.jsonl'
HEADERS = {'User-Agent': 'Mozilla/5.0 (compatible; CSC575-IIR-Crawler/1.0; +educational project)'}

@dataclass
class Document:
    id: int
    source: str
    url: str
    title: str
    prose_text: str
    code_text: str
    score: int
    is_accepted: bool
_robots_cache: dict = {}

def _get_robots(base_url: str) -> urllib.robotparser.RobotFileParser:
    if base_url not in _robots_cache:
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(urljoin(base_url, '/robots.txt'))
        try:
            rp.read()
        except Exception:
            pass
        _robots_cache[base_url] = rp
    return _robots_cache[base_url]

# Respect robots.txt before downloading any page.
def allowed_by_robots(url: str) -> bool:
    parsed = urlparse(url)
    base = f'{parsed.scheme}://{parsed.netloc}'
    rp = _get_robots(base)
    if rp.can_fetch(HEADERS['User-Agent'], url):
        return True
    if rp.can_fetch('*', url):
        return True
    return False

# Keep the crawl focused on useful HTML pages from approved documentation domains.
def is_allowed_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in ('http', 'https'):
        return False
    if parsed.netloc not in ALLOWED_DOMAINS:
        return False
    if parsed.netloc == 'docs.python.org':
        if re.match('^/(2|3\\.[0-9]+|2\\.[0-9]+)/', parsed.path):
            return False
    if parsed.netloc == 'developer.mozilla.org':
        if any((parsed.path.startswith(p) for p in MDN_SKIP_PREFIXES)):
            return False
    if parsed.netloc == 'docs.python.org':
        boring_segments = {'numeric', 'filesys', 'allos', 'ipc', 'internet', 'markup', 'mm', 'crypto', 'archiving', 'binary', 'datatypes', 'functional', 'text', 'persistence', 'fileformats', 'concurrency', 'cmdlinelibs', 'i18n', 'netdata', 'tk', 'intro', 'whatnow', 'appendix', 'debug', 'development', 'mm', 'xml', 'concurrent'}
        segment = parsed.path.rstrip('/').split('/')[-1].replace('.html', '')
        if segment in boring_segments:
            return False
    if re.search('\\.(pdf|zip|png|jpg|jpeg|svg|css|js|json|xml|ico|woff|woff2)$', parsed.path, re.I):
        return False
    return True

# Extract title, prose, code blocks, and outgoing links from one HTML page.
def parse_page(url: str, html: str) -> tuple:
    soup = BeautifulSoup(html, 'html.parser')
    h1 = soup.find('h1')
    title_tag = soup.find('title')
    if h1:
        title = h1.get_text(strip=True)
    elif title_tag:
        title = title_tag.get_text(strip=True)
    else:
        title = url
    title = title.encode('utf-8', 'ignore').decode('utf-8')
    title = re.sub('[^\\x00-\\x7F¶§]+', '', title)
    title = title.replace('¶', '').replace('§', '').strip()
    main = soup.find('div', {'class': re.compile('\\b(body|content|main|article|document)\\b', re.I)}) or soup.find('main') or soup.find('article') or soup.body or soup
    # Code is stored separately so code identifiers can be tokenized differently later.
    code_parts = []
    for tag in main.find_all(['pre', 'code']):
        text = tag.get_text()
        if text.strip():
            code_parts.append(text)
        tag.decompose()
    code_text = '\n'.join(code_parts).strip()
    prose_tags = ['p', 'li', 'td', 'th', 'dt', 'dd', 'h1', 'h2', 'h3', 'h4', 'h5']
    prose_parts = []
    for tag in main.find_all(prose_tags):
        text = tag.get_text(' ', strip=True)
        if text:
            prose_parts.append(text)
    prose_text = ' '.join(prose_parts).strip()
    prose_text = re.sub('Â', '', prose_text)
    prose_text = re.sub('[^\\x00-\\x7F]+', ' ', prose_text).strip()
    links = []
    for a in soup.find_all('a', href=True):
        href = a['href']
        full_url = urljoin(url, href)
        full_url, _ = urldefrag(full_url)
        if is_allowed_url(full_url):
            links.append(full_url)
    return (title, prose_text, code_text, links)

# Breadth-first crawl that writes each accepted page as one JSONL document.
def crawl(start_urls: list=START_URLS, max_pages: int=MAX_PAGES, output: str=OUTPUT_FILE) -> int:
    visited: set = set()
    queue = deque(start_urls)
    doc_id = 0
    saved = 0
    session = requests.Session()
    session.headers.update(HEADERS)
    log.info(f'Starting crawl | max_pages={max_pages} | output={output}')
    log.info(f'Start URLs: {start_urls}')
    with open(output, 'w', encoding='utf-8') as fh:
        while queue and doc_id < max_pages:
            url = queue.popleft()
            if url in visited:
                continue
            visited.add(url)
            if not allowed_by_robots(url):
                log.debug(f'robots.txt blocked: {url}')
                continue
            try:
                resp = session.get(url, timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
            except requests.RequestException as exc:
                log.warning(f'Fetch failed ({url}): {exc}')
                time.sleep(REQUEST_DELAY)
                continue
            content_type = resp.headers.get('Content-Type', '')
            if 'html' not in content_type:
                continue
            source = 'python_docs' if 'python.org' in urlparse(url).netloc else 'mdn'
            title, prose_text, code_text, links = parse_page(url, resp.text)
            # Remove thin pages so the index contains searchable learning content.
            if len(prose_text) < 100 and len(code_text) < 50:
                log.debug(f'Thin page skipped: {url}')
            elif any((junk in title.lower() for junk in JUNK_TITLES)):
                log.info(f"Junk title skipped: '{title}'")
            else:
                doc = Document(id=doc_id, source=source, url=url, title=title, prose_text=prose_text, code_text=code_text, score=0, is_accepted=False)
                fh.write(json.dumps(asdict(doc), ensure_ascii=False) + '\n')
                fh.flush()
                saved += 1
                doc_id += 1
                log.info(f'[{saved:>4}] {source:12} | {title[:55]:<55} | {url}')
            for link in links:
                if link not in visited:
                    queue.append(link)
            time.sleep(REQUEST_DELAY)
    log.info(f"\nCrawl complete. {saved} documents written to '{output}'")
    return saved
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='IIR focused web crawler — CSC 575')
    ap.add_argument('--max', type=int, default=MAX_PAGES, help='Max pages to crawl (default 2000)')
    ap.add_argument('--output', type=str, default=OUTPUT_FILE, help='Output .jsonl file (default docs.jsonl)')
    ap.add_argument('--url', type=str, nargs='+', help='Override start URLs')
    args = ap.parse_args()
    start = args.url if args.url else START_URLS
    crawl(start_urls=start, max_pages=args.max, output=args.output)