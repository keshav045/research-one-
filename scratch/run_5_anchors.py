import sys, os
sys.path.insert(0, os.getcwd())
import asyncio
import json
from backend.services.query_planner import plan_research_queries
from backend.services.paper_retrieval import retrieve_papers
from backend.services.ranker import select_anchor_paper, normalize_paper_title
from backend.models.schemas import ResearchDepth, ResearchSource

test_cases = [
    {
        "question": "Which paper proposed the Adam optimizer?",
        "expected_title": "Adam: A Method for Stochastic Optimization",
        "expected_id": "1412.6980",
        "authors": "Kingma & Ba",
    },
    {
        "question": "Who introduced generative adversarial networks?",
        "expected_title": "Generative Adversarial Networks",
        "expected_id": "1406.2661",
        "authors": "Goodfellow et al.",
    },
    {
        "question": "What makes YOLO different from Faster R-CNN?",
        "expected_title": "You Only Look Once: Unified, Real-Time Object Detection",
        "expected_id": "1506.02640",
        "authors": "Redmon et al.",
    },
    {
        "question": "What is retrieval-augmented generation and who proposed it?",
        "expected_title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "expected_id": "2005.11401",
        "authors": "Lewis et al.",
    },
    {
        "question": "Which paper introduced dropout?",
        "expected_title": "Dropout: A Simple Way to Prevent Neural Networks from Overfitting",
        "expected_id": "1207.0580 / JMLR",
        "authors": "Srivastava et al.",
    },
]

async def run_case(case):
    q = case["question"]
    print(f"\n=======================================================")
    print(f"QUESTION: {q}")
    print(f"Expected: '{case['expected_title']}' ({case['authors']}, {case['expected_id']})")
    print(f"=======================================================")
    
    # 1. Plan queries
    plan = await plan_research_queries(q)
    print("Plan queries:", plan.get("queries"))
    print("Plan title guesses:", plan.get("title_guesses"))
    
    # 2. Retrieve papers
    raw_papers = await retrieve_papers(
        q,
        depth=ResearchDepth.STANDARD,
        sources=[ResearchSource.ARXIV, ResearchSource.SEMANTIC_SCHOLAR],
        custom_queries=plan.get("queries"),
        title_guesses=plan.get("title_guesses"),
    )
    print(f"Retrieved {len(raw_papers)} candidate papers.")
    
    # Check if expected paper is in retrieved candidates
    exp_norm = normalize_paper_title(case["expected_title"])
    found_target = None
    for p in raw_papers:
        p_norm = normalize_paper_title(p.title)
        doi_lower = (p.doi or "").lower()
        pdf_lower = (p.pdfUrl or "").lower()
        if (case["expected_id"].split()[0].lower() in doi_lower or 
            case["expected_id"].split()[0].lower() in pdf_lower or 
            exp_norm in p_norm or 
            p_norm in exp_norm):
            found_target = p
            break
            
    print(f"Is real target paper retrieved?: {'YES' if found_target else 'NO'}")
    if found_target:
        print(f"  Target record: '{found_target.title}' | DOI: {found_target.doi} | Source: {found_target.source}")
        
    # 3. Anchor selection
    anchor, updated_papers, debug = select_anchor_paper(raw_papers, plan.get("title_guesses", []))
    if anchor:
        print(f"Anchor Chosen: '{anchor.title}'")
        print(f"  ID/DOI: {anchor.doi or anchor.id}")
        print(f"  Match Type: {debug.get('match_type')}")
        print(f"  Matched Title Guess: '{debug.get('matched_guess')}'")
    else:
        print(f"Anchor Chosen: None (Reason: {debug.get('reason')})")

async def main():
    for case in test_cases:
        await run_case(case)

if __name__ == "__main__":
    asyncio.run(main())
