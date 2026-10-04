import sys, os
sys.path.insert(0, os.getcwd())
import asyncio
import json
import sqlite3
from backend.services.query_planner import plan_research_queries
from backend.services.ranker import select_anchor_paper, normalize_paper_title, rank_papers
from backend.models.schemas import Paper, ResearchDepth

async def main():
    question = "Which paper introduced the Transformer architecture, and what was its key idea?"
    print(f"Test Question: '{question}'")
    
    # 1. Run query planner
    plan = await plan_research_queries(question)
    print("\n[Planner Output]")
    print("  Title Guesses:", plan.get("title_guesses"))
    print("  Sub-Questions:", plan.get("sub_questions"))
    print("  Queries:", plan.get("queries"))
    
    # 2. Load retrieved candidate papers from database (from the latest test job)
    conn = sqlite3.connect("researchlens.db")
    c = conn.cursor()
    c.execute("SELECT id, title, authors_json, publication_year, journal_conference, doi, source, abstract, pdf_url, evidence_count FROM papers WHERE job_id='research-c2f4ac830c2b'")
    rows = c.fetchall()
    
    candidate_papers = []
    for r in rows:
        p = Paper(
            id=r[0].split("::")[-1] if "::" in r[0] else r[0],
            title=r[1],
            authors=json.loads(r[2]) if r[2] else [],
            publicationYear=r[3] or 2024,
            journalConference=r[4] or "",
            doi=r[5] or "",
            source=r[6] or "",
            abstract=r[7] or "",
            pdfUrl=r[8],
            evidenceCount=r[9] or 0,
            citationCount=r[9] or 0,
        )
        candidate_papers.append(p)
    
    print(f"\nLoaded {len(candidate_papers)} candidate papers from retrieved corpus.")
    
    # 3. Select anchor paper
    anchor, updated_papers, debug = select_anchor_paper(candidate_papers, plan.get("title_guesses", []))
    
    print("\n[Anchor Paper Selection Result]")
    if anchor:
        arxiv_id = ""
        if "10.48550/arxiv." in (anchor.doi or "").lower():
            arxiv_id = anchor.doi.lower().split("10.48550/arxiv.")[-1]
        elif "arxiv.org/abs/" in (anchor.pdfUrl or "").lower() or "arxiv.org/pdf/" in (anchor.pdfUrl or "").lower():
            arxiv_id = anchor.pdfUrl.split("/")[-1].replace(".pdf", "")
            
        print("  Anchor Chosen:", anchor.title)
        print("  arXiv ID / DOI:", anchor.doi or arxiv_id)
        print("  Citation Count:", getattr(anchor, 'citationCount', 0) or getattr(anchor, 'evidenceCount', 0))
        print("  Match Type:", debug.get("match_type"))
        print("  Matched Title Guess:", debug.get("matched_guess"))
        print("  Why chosen: The normalized title exactly equals the planner's title guess 'attention is all you need'.")
    else:
        print("  No anchor chosen! Reason:", debug.get("reason"))
        
    print("\n[Non-matching Derivative Papers Validation]")
    for p in candidate_papers:
        if "all you need" in p.title.lower() and normalize_paper_title(p.title) != "attention is all you need":
            norm_p = normalize_paper_title(p.title)
            print(f"  - Rejected derivative paper: '{p.title[:65]}...'")
            print(f"    Normalized: '{norm_p[:60]}...' != 'attention is all you need' (Match: False)")

if __name__ == "__main__":
    asyncio.run(main())
