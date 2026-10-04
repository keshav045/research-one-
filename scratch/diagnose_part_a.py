import sys, os
sys.path.insert(0, os.getcwd())
import asyncio
import json
from backend.services.query_planner import plan_research_queries
from backend.services.paper_retrieval import retrieve_papers, get_and_clear_s2_call_records, _record_s2_call
from backend.models.schemas import ResearchDepth, ResearchSource

questions = [
    "Which paper introduced the Transformer architecture, and what was its key idea?",
    "Who introduced generative adversarial networks?",
    "Which paper proposed the Adam optimizer?",
    "What is retrieval-augmented generation and who proposed it?",
    "Which paper introduced dropout?"
]

async def check():
    for q in questions:
        print(f"\n=======================================================")
        print(f"QUESTION: {q}")
        print(f"=======================================================")
        get_and_clear_s2_call_records()
        plan = await plan_research_queries(q)
        print("Plan queries:", plan.get("queries"))
        print("Plan title guesses:", plan.get("title_guesses"))
        papers = await retrieve_papers(
            q,
            depth=ResearchDepth.STANDARD,
            sources=[ResearchSource.ARXIV, ResearchSource.SEMANTIC_SCHOLAR],
            custom_queries=plan.get("queries"),
            title_guesses=plan.get("title_guesses")
        )
        records = get_and_clear_s2_call_records()
        print(f"Total candidate papers retrieved: {len(papers)}")
        print(f"S2 Call History ({len(records)} calls recorded):")
        for r in records:
            print(f"  Query: '{r['query']}' | Status: {r['status_code']} | Count: {r['papers_count']} | Error: {r['error']}")

if __name__ == "__main__":
    asyncio.run(check())
