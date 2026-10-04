import sys
import io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
from backend.services.paper_retrieval import retrieve_papers
from backend.services.query_planner import plan_research_queries
from backend.services.ranker import select_anchor_paper, rank_papers
from backend.services.pdf_extractor import extract_papers_batch
from backend.services.vector_store import VectorStore
from backend.services.evidence import extract_and_verify_evidence
from backend.models.schemas import ResearchDepth, ResearchSource
from backend.config import settings

async def main():
    q = "Which paper introduced the Transformer architecture, and what was its key idea?"
    plan = await plan_research_queries(q)
    sub_qs = plan["sub_questions"]
    
    papers = await retrieve_papers(
        q,
        ResearchDepth.STANDARD,
        [ResearchSource.ARXIV],
        custom_queries=plan["queries"],
        title_guesses=plan["title_guesses"],
    )
    anchor, papers, debug = select_anchor_paper(papers, plan["title_guesses"])
    ranked = rank_papers(q, papers, plan["question_type"], plan["title_guesses"], ResearchDepth.STANDARD, anchor)
    enriched = await extract_papers_batch(ranked[:6], max_workers=2)
    vs = VectorStore()
    vs.build(enriched)
    
    anchor_id = anchor.id if anchor else None
    
    print("=" * 75)
    print("STEP 5 END-TO-END VALIDATION RUN")
    print(f"Question: \"{q}\"")
    print(f"Anchor Paper: \"{anchor.title if anchor else None}\" (ID: {anchor_id})")
    print(f"Sub-Questions: {sub_qs}")
    print(f"Thresholds: RELEVANCE_THRESHOLD = {settings.RELEVANCE_THRESHOLD}, NLI_ENTAIL_THRESHOLD = {settings.NLI_ENTAIL_THRESHOLD}")
    print("=" * 75)
    
    claims, citations, rejected_claims, removal_counts = extract_and_verify_evidence(
        question=q,
        sub_questions=sub_qs,
        papers=enriched,
        vector_store=vs,
        max_citations=15,
        anchor_paper_id=anchor_id,
    )

    print(f"\n>>> EVIDENCE FILTER REMOVALS COUNTS:")
    for k, v in removal_counts.items():
        print(f"  {k}: {v}")
    
    print(f"\n>>> VERIFIED CITATIONS / CLAIMS ({len(citations)}):")
    for c in citations:
        print(f"Badge {c.badgeNumber}: [VERIFIED] (Page {c.page}, {c.paperTitle})")
        print(f"  Claim: \"{c.claim}\"")
        print(f"  Relevance Score: {c.extractionConfidence:.4f} | Entailment Score: {c.entailmentScore:.4f}\n")
        
    print(f"\n>>> REJECTED CLAIMS ({len(rejected_claims)}):")
    for r in rejected_claims:
        print(f"  [REJECTED: {r['reason']}] (Sub-Q: '{r['sub_question']}')")
        print(f"    Claim: \"{r['claim']}\"")
        print(f"    Relevance: {r['relevance_score']:.4f} | Check: {r.get('check_label', 'source match')}\n")

if __name__ == "__main__":
    asyncio.run(main())
