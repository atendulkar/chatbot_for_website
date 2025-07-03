import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time

BASE_URL = "https://content.naic.org/"
model = SentenceTransformer("all-MiniLM-L6-v2")

def is_valid_internal_link(href):
    if not href:
        return False
    parsed = urlparse(href)
    if parsed.scheme in ['http', 'https'] and parsed.netloc and parsed.netloc != urlparse(BASE_URL).netloc:
        return False
    return True

def crawl_site(base_url=BASE_URL, max_pages=50):
    visited, to_visit = set(), [base_url]
    all_links = []

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        try:
            print(f"Crawling: {url}")
            visited.add(url)
            resp = requests.get(url, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")
            all_links.append(url)

            for a in soup.find_all("a", href=True):
                full_url = urljoin(url, a['href'])
                if is_valid_internal_link(full_url) and full_url not in visited and full_url not in to_visit:
                    to_visit.append(full_url)

        except Exception as e:
            print(f"Error crawling {url}: {e}")
        time.sleep(0.5)
    return all_links

def chunk_text(text, chunk_size=4):
    """Groups list of sentences into chunks of N sentences."""
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
    chunks = ['. '.join(sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size)]
    return chunks

def extract_chunks_from_url(url, chunk_size=4):
    try:
        resp = requests.get(url, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        paragraphs = soup.find_all('p')
        text = ' '.join(p.get_text(strip=True) for p in paragraphs)
        return chunk_text(text, chunk_size)
    except Exception as e:
        print(f"Scrape error on {url}: {e}")
        return []

def build_knowledge_base(max_pages=50, chunk_size=4):
    kb = []
    urls = crawl_site(BASE_URL, max_pages=max_pages)
    for url in urls:
        chunks = extract_chunks_from_url(url, chunk_size)
        kb.extend(chunks)
    embeddings = model.encode(kb)
    return kb, embeddings

def generate_response_vector(query, kb, kb_embeddings, top_n=2):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, kb_embeddings)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]
    return '\n\n'.join(kb[i] for i in top_indices)
