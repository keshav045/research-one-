import sys, os
sys.path.insert(0, os.getcwd())
import sqlite3
import json
import httpx
import asyncio
from backend.config import settings
from backend.services.ranker import filter_and_deduplicate
from backend.models.schemas import Paper

def main():
    conn = sqlite3.connect("researchlens.db")
    c = conn.cursor()
    c.execute("SELECT id, title, source, evidence_count FROM papers WHERE job_id='research-c2f4ac830c2b'")
    rows = c.fetchall()
    
    sources = {}
    for r in rows:
        sources[r[2]] = sources.get(r[2], 0) + 1
    print("1. Papers count by source in latest job:")
    print("  ", sources)
    
    print("\n2. Checking Semantic Scholar API HTTP responses:")
    query = "Attention Is All You Need Vaswani"
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit=3&fields=paperId,title,citationCount"
    headers = {}
    s2_key = settings.SEMANTIC_SCHOLAR_API_KEY.strip() if settings.SEMANTIC_SCHOLAR_API_KEY else ""
    if s2_key:
        headers["x-api-key"] = s2_key
    
    try:
        resp = httpx.get(url, headers=headers, timeout=10.0)
        print(f"   HTTP Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"   Returned total matches: {data.get('total')}")
            for p in data.get("data", [])[:2]:
                print(f"   - {p.get('title')} | Citations: {p.get('citationCount')}")
        else:
            print(f"   Response text: {resp.text[:200]}")
    except Exception as e:
        print(f"   Request error: {e}")
        
    print("\n3. Is SEMANTIC_SCHOLAR_API_KEY set?:", "yes" if bool(s2_key) else "no")
    
    print("\n4. Checking deduplication behavior when arXiv and S2 paper are same work:")
    arxiv_p = Paper(
        id="arxiv-1",
        title="Attention Is All You Need",
        authors=["Vaswani"],
        publicationYear=2017,
        journalConference="NeurIPS",
        doi="10.48550/arXiv.1706.03762",
        source="arXiv",
        abstract="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
        evidenceCount=0,
        citationCount=0,
    )
    s2_p = Paper(
        id="s2-1",
        title="Attention Is All You Need",
        authors=["Vaswani"],
        publicationYear=2017,
        journalConference="NeurIPS",
        doi="10.48550/arXiv.1706.03762",
        source="Semantic Scholar",
        abstract="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
        evidenceCount=120000,
        citationCount=120000,
    )
    
    # What does filter_and_deduplicate keep when arXiv comes first?
    deduped = filter_and_deduplicate([arxiv_p, s2_p])
    print(f"   Input: [arXiv (cites=0), S2 (cites=120000)]")
    print(f"   Deduplicated output kept: {deduped[0].source} (ID: {deduped[0].id})")
    print(f"   Deduplicated citation count: evidenceCount={deduped[0].evidenceCount}, citationCount={getattr(deduped[0], 'citationCount', 0)}")
    print(f"   Result: Dedupe keeps the first record (arXiv) with 0 citations and drops the S2 record.")

if __name__ == "__main__":
    main()
