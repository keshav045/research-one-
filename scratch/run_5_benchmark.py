import asyncio
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, ".")

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Benchmark")

from backend.config import settings
from backend.models.schemas import ResearchDepth, ResearchSource, Paper
from backend.services.query_planner import plan_research_queries
from backend.services.paper_retrieval import (
    retrieve_candidate_papers,
    enrich_papers_with_s2,
    get_and_clear_s2_call_records,
    get_and_clear_retrieval_errors,
)
from backend.services.ranker import (
    filter_and_deduplicate,
    select_anchor_paper,
    rank_papers,
    normalize_paper_title,
    get_reranker,
)
from backend.services.pdf_extractor import extract_papers_batch
from backend.services.vector_store import VectorStore
from backend.services.evidence import (
    extract_and_verify_evidence,
    clean_hyphenated_breaks,
)

QUESTIONS = [
    {
        "topic": "Transformer",
        "question": "Which paper introduced the Transformer architecture, and what was its key idea?",
        "expected_real": "1706.03762",
        "expected_title": "attention is all you need",
    },
    {
        "topic": "GAN",
        "question": "Who introduced generative adversarial networks?",
        "expected_real": "1406.2661",
        "expected_title": "generative adversarial nets",
    },
    {
        "topic": "Adam",
        "question": "Which paper proposed the Adam optimizer?",
        "expected_real": "1412.6980",
        "expected_title": "adam a method for stochastic optimization",
    },
    {
        "topic": "RAG",
        "question": "What is retrieval-augmented generation and who proposed it?",
        "expected_real": "2005.11401",
        "expected_title": "retrieval augmented generation for knowledge intensive nlp tasks",
    },
    {
        "topic": "Dropout",
        "question": "Which paper introduced dropout?",
        "expected_real": "1207.0580",
        "expected_title": "dropout",
    },
]


async def run_benchmark():
    reranker = get_reranker()
    print("CrossEncoder loaded.")
    sources = [ResearchSource.ARXIV, ResearchSource.SEMANTIC_SCHOLAR]
    depth = ResearchDepth.STANDARD

    results = []

    for item in QUESTIONS:
        topic = item["topic"]
        q = item["question"]
        print(f"\n{'='*70}\nRUNNING BENCHMARK FOR: {topic} ('{q}')\n{'='*70}")

        # 1. Query planning
        plan = await plan_research_queries(q)
        q_type = plan.get("question_type", "factual_lookup")
        queries = plan.get("queries", [q])
        title_guesses = plan.get("title_guesses", [])
        sub_questions = plan.get("sub_questions", [q])

        print(f"Plan: type={q_type}, title_guesses={title_guesses}, sub_questions={sub_questions}")

        # Clear any prior records
        get_and_clear_s2_call_records()
        get_and_clear_retrieval_errors()

        # 2. Retrieval
        raw_candidates, _ = await retrieve_candidate_papers(q, depth, sources, plan={
            "question_type": q_type,
            "queries": queries,
            "title_guesses": title_guesses,
            "sub_questions": sub_questions,
        })
        s2_records = get_and_clear_s2_call_records()
        r_errors = get_and_clear_retrieval_errors()

        arxiv_count = sum(1 for p in raw_candidates if p.source == "arXiv")
        s2_count = sum(1 for p in raw_candidates if p.source == "Semantic Scholar")

        # 3. Deduplication with metadata merging
        deduped_candidates = filter_and_deduplicate(raw_candidates)

        # 4. Batch Citation Enrichment via Semantic Scholar
        enriched_candidates = await enrich_papers_with_s2(deduped_candidates)
        enriched_with_cites_count = sum(
            1 for p in enriched_candidates if (getattr(p, "citationCount", 0) or 0) > 0
        )

        # 5. Anchor Paper Selection
        anchor_paper = None
        anchor_debug = {}
        if q_type == "factual_lookup" and title_guesses:
            anchor_paper, enriched_candidates, anchor_debug = select_anchor_paper(enriched_candidates, title_guesses)

        # 6. Ranking (Cut to top N happens AFTER enrichment)
        ranked_papers = rank_papers(
            question=q,
            papers=enriched_candidates,
            question_type=q_type,
            title_guesses=title_guesses,
            depth=depth,
            anchor_paper=anchor_paper,
        )

        # Check if real paper is among final ranked candidates (and at what rank)
        real_paper_in_final = False
        real_rank = None
        matching_cand = None
        for idx, p in enumerate(ranked_papers):
            p_norm = normalize_paper_title(p.title)
            if item["expected_real"] in p.id.lower() or item["expected_real"] in (p.doi or "").lower() or item["expected_title"] in p_norm:
                real_paper_in_final = True
                real_rank = idx + 1
                matching_cand = p
                break

        # If not in top ranked, check if it was in enriched candidates before cut
        real_in_candidates_uncut = False
        uncut_pos = None
        if not real_paper_in_final:
            for idx, p in enumerate(enriched_candidates):
                p_norm = normalize_paper_title(p.title)
                if item["expected_real"] in p.id.lower() or item["expected_real"] in (p.doi or "").lower() or item["expected_title"] in p_norm:
                    real_in_candidates_uncut = True
                    uncut_pos = idx + 1
                    matching_cand = p
                    break

        # 7. Extract text for top papers (up to 4 papers to run efficiently)
        eval_papers = ranked_papers[:4]
        if anchor_paper and not any(p.id == anchor_paper.id for p in eval_papers):
            eval_papers.insert(0, anchor_paper)

        print(f"Extracting passages for {len(eval_papers)} papers...")
        extracted_papers = await extract_papers_batch(eval_papers, max_workers=2)
        passages_total = sum(len(p.passages or []) for p in extracted_papers)

        # 8. Vector Index
        vstore = VectorStore()
        if passages_total > 0:
            vstore.build(extracted_papers)

        # 9. Top 3 passages for first sub-question
        first_sub_q = sub_questions[0] if sub_questions else q
        top_3_passages = []

        if passages_total > 0 and vstore.records:
            faiss_cands = vstore.search(first_sub_q, top_k=30)
            anchor_id = anchor_paper.id if anchor_paper else None
            chosen_records = []

            if anchor_id:
                anchor_records = [r for r in vstore.records if r.paper_id == anchor_id]
                if anchor_records:
                    if reranker is not None:
                        pairs = [[first_sub_q, clean_hyphenated_breaks(r.passage.text)] for r in anchor_records]
                        scores = reranker.predict(pairs)
                        anchor_scored = [(r, float(s)) for r, s in zip(anchor_records, scores)]
                    else:
                        anchor_scored = [(r, 0.5) for r in anchor_records]
                    anchor_scored.sort(key=lambda x: x[1], reverse=True)
                    relevant_anchor = [it for it in anchor_scored if it[1] >= -1.5]
                    if len(relevant_anchor) >= 3:
                        chosen_records = anchor_scored[:3]
                    else:
                        chosen_records = list(relevant_anchor)
                        other_cands = [r for r, _ in faiss_cands if r.paper_id != anchor_id]
                        if other_cands:
                            if reranker is not None:
                                other_pairs = [[first_sub_q, clean_hyphenated_breaks(r.passage.text)] for r in other_cands]
                                other_scores = reranker.predict(other_pairs)
                                other_scored = [(r, float(s)) for r, s in zip(other_cands, other_scores)]
                            else:
                                other_scored = [(r, sc) for r, sc in faiss_cands if r.paper_id != anchor_id]
                            other_scored.sort(key=lambda x: x[1], reverse=True)
                            for it in other_scored:
                                if len(chosen_records) >= 3:
                                    break
                                if not any(it[0].passage.id == cp[0].passage.id for cp in chosen_records):
                                    chosen_records.append(it)
            else:
                cand_records = [r for r, _ in faiss_cands]
                if cand_records:
                    if reranker is not None:
                        pairs = [[first_sub_q, clean_hyphenated_breaks(r.passage.text)] for r in cand_records]
                        scores = reranker.predict(pairs)
                        scored = [(r, float(s)) for r, s in zip(cand_records, scores)]
                    else:
                        scored = list(faiss_cands)
                    scored.sort(key=lambda x: x[1], reverse=True)
                    chosen_records = scored[:3]

            for rec, sc in chosen_records:
                p_obj = next((p for p in eval_papers if p.id == rec.paper_id), None)
                p_title = p_obj.title if p_obj else rec.paper_id
                snippet = " ".join(clean_hyphenated_breaks(rec.passage.text).split())[:100]
                top_3_passages.append({
                    "paper_title": p_title,
                    "page": rec.passage.page,
                    "rerank_score": round(sc, 3),
                    "snippet": snippet,
                })

        res_entry = {
            "topic": topic,
            "question": q,
            "papers_per_source": {"arXiv": arxiv_count, "Semantic Scholar": s2_count},
            "total_deduped": len(deduped_candidates),
            "enriched_with_citations": enriched_with_cites_count,
            "real_paper_in_final": real_paper_in_final,
            "real_rank": real_rank,
            "real_in_candidates_uncut": real_in_candidates_uncut,
            "real_paper_title": matching_cand.title if matching_cand else "Missing",
            "anchor_title": anchor_paper.title if anchor_paper else "None",
            "anchor_citations": getattr(anchor_paper, "citationCount", 0) if anchor_paper else "N/A",
            "first_sub_question": first_sub_q,
            "top_3_passages": top_3_passages,
        }
        results.append(res_entry)
        print(f"Finished {topic}: real_in_final={real_paper_in_final} (rank {real_rank}), anchor={res_entry['anchor_title']} ({res_entry['anchor_citations']} cites)")

    with open("scratch/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 90)
    print("BENCHMARK SUMMARY TABLE")
    print("=" * 90)
    print(f"| Question Topic | Papers per Source (arXiv / S2) | Total Deduped | Enriched with Citations | Real Paper in Final? (Rank) | Anchor Paper & Citations | Top 3 Passages' Paper Titles (Sub-Q 1) |")
    print(f"|---|---|---|---|---|---|---|")
    for r in results:
        src = f"arXiv: {r['papers_per_source']['arXiv']} / S2: {r['papers_per_source']['Semantic Scholar']}"
        real_status = f"Yes (Rank #{r['real_rank']})" if r['real_paper_in_final'] else ("In candidates (not top 12)" if r['real_in_candidates_uncut'] else "Missing")
        anchor_info = f"{r['anchor_title'][:35]} ({r['anchor_citations']:,} cites)" if isinstance(r['anchor_citations'], int) else f"{r['anchor_title']}"
        top_titles = "<br>".join([f"• {p['paper_title'][:35]} (p.{p['page']})" for p in r['top_3_passages']]) if r['top_3_passages'] else "None"
        print(f"| **{r['topic']}** | {src} | {r['total_deduped']} | {r['enriched_with_citations']} | {real_status} | {anchor_info} | {top_titles} |")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
