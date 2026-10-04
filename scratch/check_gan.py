import sys, os
sys.path.insert(0, os.getcwd())
import asyncio
from backend.services.query_planner import plan_research_queries
from backend.services.paper_retrieval import retrieve_papers, _fetch_arxiv_query
from backend.services.ranker import filter_and_deduplicate, rank_papers, select_anchor_paper, normalize_paper_title
from backend.models.schemas import ResearchDepth, ResearchSource

async def test_gan():
    q = "Who introduced generative adversarial networks?"
    plan = await plan_research_queries(q)
    print("Plan:", plan)
    
    # Check arXiv raw query for generative adversarial networks
    arxiv_papers = await _fetch_arxiv_query("all:generative adversarial networks", max_results=50)
    print(f"Direct arXiv query returned {len(arxiv_papers)} papers.")
    found_1406 = [p for p in arxiv_papers if "1406.2661" in (p.doi or "") or "1406.2661" in (p.pdfUrl or "") or "generative adversarial nets" in p.title.lower()]
    print(f"Found Goodfellow 1406.2661 in direct arXiv query? {len(found_1406)}")
    if found_1406:
        print("Goodfellow paper:", found_1406[0].title, found_1406[0].doi)

    # Now retrieve using retrieve_papers
    raw_papers = await retrieve_papers(
        q,
        depth=ResearchDepth.STANDARD,
        sources=[ResearchSource.ARXIV, ResearchSource.SEMANTIC_SCHOLAR],
        custom_queries=plan.get("queries"),
        title_guesses=plan.get("title_guesses")
    )
    print(f"Raw candidate papers count: {len(raw_papers)}")
    
    # Check if 1406.2661 is in raw_papers
    raw_match = [p for p in raw_papers if "1406.2661" in (p.doi or "") or "1406.2661" in (p.pdfUrl or "") or "generative adversarial nets" in p.title.lower()]
    print(f"Is 1406.2661 in RAW retrieved list?: {len(raw_match) > 0}")
    if raw_match:
        print(f"Raw match title: {raw_match[0].title}")

    # Dedupe
    deduped = filter_and_deduplicate(raw_papers)
    print(f"Deduped count: {len(deduped)}")
    dedupe_match = [p for p in deduped if "1406.2661" in (p.doi or "") or "1406.2661" in (p.pdfUrl or "") or "generative adversarial nets" in p.title.lower()]
    print(f"Did it survive dedupe?: {len(dedupe_match) > 0}")

    # Rank
    ranked = rank_papers(q, deduped)
    rank_pos = None
    for idx, p in enumerate(ranked):
        if "1406.2661" in (p.doi or "") or "1406.2661" in (p.pdfUrl or "") or "generative adversarial nets" in p.title.lower():
            rank_pos = idx + 1
            break
    print(f"Ranker position: {rank_pos}")
    
    # Print top 5 ranked
    print("Top 5 ranked:")
    for idx, p in enumerate(ranked[:5]):
        print(f"  #{idx+1}: {p.title} (Source: {p.source}, Cites: {p.citationCount}, DOI: {p.doi})")

if __name__ == "__main__":
    asyncio.run(test_gan())
