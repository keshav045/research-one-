import sys
import io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import re
from backend.services.paper_retrieval import retrieve_papers
from backend.services.query_planner import plan_research_queries
from backend.services.ranker import select_anchor_paper, rank_papers, get_reranker
from backend.services.pdf_extractor import extract_papers_batch
from backend.services.vector_store import VectorStore
from backend.models.schemas import ResearchDepth, ResearchSource

def clean_hyphenated_breaks(text: str) -> str:
    return re.sub(r"\b([a-zA-Z]{2,})-\s+([a-zA-Z]{2,})\b", r"\1\2", text)

from backend.services.evidence import extract_and_verify_evidence, clean_hyphenated_breaks

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
    
    reranker = get_reranker()
    anchor_id = anchor.id if anchor else None
    
    print("=" * 70)
    print("TEST QUESTION:", q)
    print("ANCHOR PAPER:", anchor.title if anchor else None, f"(ID: {anchor_id})")
    print("SUB-QUESTIONS:", sub_qs)
    print("=" * 70)
    
    for sub_q in sub_qs:
        print(f"\n>>> SUB-QUESTION: \"{sub_q}\"")
        faiss_candidates = vs.search(sub_q, top_k=30)
        
        chosen_passages = []
        if anchor_id:
            anchor_records = [r for r in vs.records if r.paper_id == anchor_id]
            if anchor_records:
                pairs = [[sub_q, clean_hyphenated_breaks(r.passage.text)] for r in anchor_records]
                scores = reranker.predict(pairs)
                anchor_scored = [(r, float(s)) for r, s in zip(anchor_records, scores)]
                anchor_scored.sort(key=lambda x: x[1], reverse=True)
                relevant_anchor = [item for item in anchor_scored if item[1] >= -1.5]
                if len(relevant_anchor) >= 3:
                    chosen_passages = anchor_scored[:5]
                else:
                    chosen_passages = list(relevant_anchor)
                    other_candidates = [r for r, _ in faiss_candidates if r.paper_id != anchor_id]
                    if other_candidates:
                        other_pairs = [[sub_q, clean_hyphenated_breaks(r.passage.text)] for r in other_candidates]
                        other_scores = reranker.predict(other_pairs)
                        other_scored = [(r, float(s)) for r, s in zip(other_candidates, other_scores)]
                        other_scored.sort(key=lambda x: x[1], reverse=True)
                        for item in other_scored:
                            if len(chosen_passages) >= 5:
                                break
                            if not any(item[0].passage.id == cp[0].passage.id for cp in chosen_passages):
                                chosen_passages.append(item)
        else:
            cand_records = [r for r, _ in faiss_candidates]
            pairs = [[sub_q, clean_hyphenated_breaks(r.passage.text)] for r in cand_records]
            scores = reranker.predict(pairs)
            scored = [(r, float(s)) for r, s in zip(cand_records, scores)]
            scored.sort(key=lambda x: x[1], reverse=True)
            chosen_passages = scored[:5]
            
        print(f"Top 5 passages:")
        for i, (rec, sc) in enumerate(chosen_passages[:5], 1):
            cl_text = clean_hyphenated_breaks(rec.passage.text).replace("\n", " ")
            print(f"  {i}. Paper: \"{rec.paper_title}\" | Page: {rec.passage.page} | Rerank Score: {sc:.4f}")
            print(f"     Passage Text: {cl_text[:160]}...")
            
    print("\n" + "=" * 70)
    print("VERIFICATION: Does any passage from 1706.03762 mention self-attention replacing recurrence?")
    print("=" * 70)
    
    # Check all passages of 1706.03762
    anchor_recs = [r for r in vs.records if r.paper_id == anchor_id]
    recurrence_mentions = []
    for r in anchor_recs:
        txt = clean_hyphenated_breaks(r.passage.text)
        if re.search(r"replac.*recurr|dispens.*recurr|replac.*rnn|solely on attention", txt, re.IGNORECASE):
            recurrence_mentions.append((r.passage.page, txt))
            
    print(f"Total passages in 1706.03762 mentioning self-attention replacing recurrence / dispensing with recurrence: {len(recurrence_mentions)}")
    for page, txt in recurrence_mentions:
        print(f"\n[Page {page}]: \"{txt.strip()}\"")

    # Run extract_and_verify_evidence to ensure end-to-end integration works
    claims, citations = extract_and_verify_evidence(
        question=q,
        sub_questions=sub_qs,
        papers=enriched,
        vector_store=vs,
        max_citations=15,
        anchor_paper_id=anchor_id,
    )
    print("\n" + "=" * 70)
    print(f"extract_and_verify_evidence produced {len(claims)} claims and {len(citations)} citations.")
    print("=" * 70)
    for c in citations[:5]:
        print(f"Badge {c.badgeNumber}: [{c.status.value}] Paper: '{c.paperTitle}' (Page {c.page})")
        print(f"  Claim: '{c.claim}'")
        print(f"  Entailment Score: {c.entailmentScore:.3f}, Rerank Score: {c.extractionConfidence:.4f}")

if __name__ == "__main__":
    asyncio.run(main())
