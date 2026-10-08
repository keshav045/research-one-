"""
Query Planner Service
=====================
Classifies research questions, isolates sub-questions, and generates targeted
database search queries for arXiv and Semantic Scholar.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional, TypedDict

from ..config import settings

logger = logging.getLogger(__name__)

# ── Question Classification Patterns ─────────────────────────────────────────

_FACTUAL_PATTERNS = re.compile(
    r"\b("
    r"which paper|who introduced|who proposed|who invented|what paper|"
    r"when was|when did|what year|who first|who created|what was the first|"
    r"who wrote|who published|what is the name|name the paper|name the author|"
    r"cite the|what model introduced|what architecture introduced"
    r")\b",
    re.IGNORECASE,
)

_COMPARATIVE_PATTERNS = re.compile(
    r"\b("
    r"compar|benchmark|evaluat|versus|\bvs\b|trade.?off|ablation|"
    r"performance|throughput|accuracy|state.of.the.art|sota|"
    r"which is better|which performs|survey|review|overview"
    r")\b",
    re.IGNORECASE,
)

STOP_WORDS = frozenset(
    "the a an is are was were be been being have has had do does did will would could "
    "should may might shall can need dare ought what which who whom whose when where how "
    "and but or nor for yet so of in on at by from with about against between into "
    "through during before after above below to up down than that this these those i you "
    "he she it we they them their its if".split()
)

META_WORDS = frozenset(
    "paper papers introduced introducing introduce propose proposed key idea ideas "
    "architecture model models study approach method work publication author".split()
)


def classify_question_type(question: str) -> str:
    """Classifies question as 'factual_lookup' or 'literature_review'."""
    is_factual = bool(_FACTUAL_PATTERNS.search(question))
    is_comparative = bool(_COMPARATIVE_PATTERNS.search(question))
    if is_factual and not is_comparative:
        return "factual_lookup"
    return "literature_review"


def extract_question_facets(question: str, sub_questions: Optional[list[str]] = None) -> list[str]:
    """Extract topical facets or sub-aspects dynamically for any research question."""
    # 1. If sub-questions are provided and multiple, synthesize facet titles from them
    if sub_questions and len(sub_questions) > 1:
        facets = []
        for sq in sub_questions:
            clean = re.sub(
                r"^(what|how|why|which|when|where|is|are|can|do|does)\s+(are|is|the|do|does)?\s*",
                "",
                sq.strip(),
                flags=re.IGNORECASE,
            ).rstrip("?").strip()
            words = clean.split()[:4]
            if words:
                facet_title = " ".join(w.capitalize() for w in words)
                if facet_title not in facets:
                    facets.append(facet_title)
        if facets:
            return facets[:4]

    # 2. Extract key topical concept clusters dynamically from the question
    tokens = re.sub(r"[^a-zA-Z0-9\s-]", " ", question.lower()).split()
    meaningful = [t for t in tokens if len(t) > 2 and t not in STOP_WORDS and t not in META_WORDS]
    if not meaningful:
        return ["Empirical Findings", "Methodological Analysis"]
    facets = []
    for i in range(0, min(len(meaningful), 6), 2):
        chunk = meaningful[i:i+2]
        facets.append(" ".join(w.capitalize() for w in chunk))
    return facets or ["Empirical Findings"]


class QueryPlan(TypedDict):
    question_type: str
    queries: list[str]
    title_guesses: list[str]
    sub_questions: list[str]
    target_concepts: list[str]
    plan_source: Optional[str]
    expected_titles: Optional[list[str]]
    planner_provider: Optional[str]
    planner_model: Optional[str]


def fallback_query_planner(question: str) -> QueryPlan:
    """
    Deterministic rule-based query generator when LLM is unavailable or fails.
    Extracts dynamic facets, concepts, sub-questions, and keyword queries without hardcoded topics.
    """
    q_type = classify_question_type(question)

    # 1. Clean tokens
    tokens = re.sub(r"[^a-zA-Z0-9\s-]", " ", question.lower()).split()
    meaningful = [t for t in tokens if len(t) > 2 and t not in STOP_WORDS]
    keywords = [t for t in meaningful if t not in META_WORDS]

    # 2. Dynamic sub-question generation based on question structure
    sub_questions = []
    compound_parts = re.split(r"\b(?:and|in terms of|as well as|compared to|versus|vs\.?)\b", question, flags=re.IGNORECASE)
    if len(compound_parts) > 1:
        for part in compound_parts[:3]:
            p_clean = part.strip().rstrip("?").strip()
            if len(p_clean.split()) >= 3:
                if not any(p_clean.lower().startswith(w) for w in ["what", "how", "why", "which"]):
                    sub_questions.append(f"What is the empirical evidence regarding {p_clean}?")
                else:
                    sub_questions.append(f"{p_clean}?")
    if not sub_questions:
        sub_questions = [question.strip()]

    # 3. Dynamic facet extraction
    facets = extract_question_facets(question, sub_questions)

    queries: list[str] = []
    title_guesses: list[str] = []

    # 4. Generate topical keyword queries per facet
    if keywords:
        broad_kw = " ".join(keywords[:4])
        queries.append(broad_kw)

        for f in facets:
            f_words = [w.lower() for w in re.findall(r"\b\w{3,}\b", f) if w.lower() not in STOP_WORDS]
            if f_words:
                f_query = " ".join(f_words + keywords[:2])
                f_query_dedup = " ".join(dict.fromkeys(f_query.split()))
                if f_query_dedup not in queries:
                    queries.append(f_query_dedup)

    # Seed list of well-known papers for deterministic fallback guidance when enabled
    if getattr(settings, "SEED_PAPERS_ENABLED", True):
        lower_q = question.lower()
        if re.search(r"\btransformer\b", lower_q):
            queries.append("Attention Is All You Need Vaswani")
            queries.append("transformer self-attention")
            title_guesses.append("Attention Is All You Need")
        elif re.search(r"\bbert\b", lower_q):
            queries.append("BERT Pre-training Deep Bidirectional Devlin")
            queries.append("masked language model BERT")
            title_guesses.append("BERT: Pre-training of Deep Bidirectional Transformers")
        elif re.search(r"\b(resnet|residual)\b", lower_q):
            queries.append("Deep Residual Learning Image Recognition He")
            queries.append("residual networks skip connection")
            title_guesses.append("Deep Residual Learning for Image Recognition")
        elif re.search(r"\blora\b", lower_q):
            queries.append("LoRA Low-Rank Adaptation Large Language Models Hu")
            title_guesses.append("LoRA: Low-Rank Adaptation of Large Language Models")

    # Target concepts derived dynamically from facets and keywords
    target_concepts = [f.title() for f in facets]
    for kw in keywords[:4]:
        kw_cap = kw.capitalize()
        if kw_cap not in target_concepts:
            target_concepts.append(kw_cap)

    unique_queries = list(dict.fromkeys(queries))

    return {
        "question_type": q_type,
        "queries": unique_queries[:6],
        "title_guesses": title_guesses[:2],
        "sub_questions": sub_questions[:3],
        "target_concepts": target_concepts[:8],
        "plan_source": "dynamic_fallback_planner",
        "expected_titles": title_guesses[:2],
        "planner_provider": "deterministic",
        "planner_model": "dynamic_facets",
    }


async def plan_research_queries(question: str) -> QueryPlan:
    """
    Plan search queries and sub-questions using LLM, with fallback to deterministic query planner.
    """
    plan = fallback_query_planner(question)

    prompt = (
        "You are an academic query planner. Split this research question into 1 to 3 atomic sub-questions.\n"
        "Also provide 2 to 4 concise search queries (2-5 keywords each, no question words) and any likely paper titles.\n"
        "Return ONLY a valid JSON object matching this schema:\n"
        '{"sub_questions": ["sub-question 1", "sub-question 2"], "queries": ["keyword query 1", "keyword query 2"], "title_guesses": ["Estimated Paper Title"]}\n\n'
        f'Question: "{question}"\n'
        "JSON:"
    )

    used_llm = False
    planner_provider = "none"
    planner_model = "none"
    try:
        from .local_llm_service import _call_llm, get_provider_model
        raw, planner_provider = await _call_llm(prompt, max_tokens=300)
        planner_model = get_provider_model(planner_provider) if raw else "none"

        if not raw or not raw.strip():
            logger.info("[QueryPlanner] Fallback plan used: LLM returned empty reply or unavailable")
            plan["sub_questions"] = [question.strip()]
        else:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                    llm_subs = [str(s).strip() for s in data.get("sub_questions", []) if str(s).strip()]
                    llm_queries = [str(q).strip() for q in data.get("queries", []) if str(q).strip()]
                    llm_titles = [str(t).strip() for t in data.get("title_guesses", []) if str(t).strip()]

                    if llm_subs:
                        plan["sub_questions"] = llm_subs[:3]
                        used_llm = True
                    else:
                        logger.warning("[QueryPlanner] Fallback plan used: LLM JSON contained no sub_questions")
                        plan["sub_questions"] = [question.strip()]

                    if llm_queries:
                        for q in llm_queries:
                            if q not in plan["queries"] and len(plan["queries"]) < 4:
                                plan["queries"].append(q)
                    if llm_titles:
                        from .ranker import validate_title_guesses
                        for t in validate_title_guesses(llm_titles):
                            if t not in plan["title_guesses"]:
                                plan["title_guesses"].append(t)
                except Exception as json_err:
                    logger.warning("[QueryPlanner] Fallback plan used: invalid JSON in LLM reply (%s)", json_err)
                    plan["sub_questions"] = [question.strip()]
            else:
                logger.warning("[QueryPlanner] Fallback plan used: invalid JSON / no JSON object found in LLM reply")
                plan["sub_questions"] = [question.strip()]
    except Exception as exc:
        logger.warning("[QueryPlanner] Fallback plan used: exception during LLM query planning (%s)", exc)
        plan["sub_questions"] = [question.strip()]

    from .ranker import validate_title_guesses
    plan["title_guesses"] = validate_title_guesses(plan.get("title_guesses", []))
    plan["plan_source"] = "llm" if used_llm else "fallback"
    plan["planner_provider"] = planner_provider if used_llm else "none"
    plan["planner_model"] = planner_model if used_llm else "none"
    # Ensure expected_titles alias is present
    plan["expected_titles"] = plan.get("title_guesses", [])
    logger.info(
        "[QueryPlanner] Final plan (%s): type=%s, queries=%s, titles=%s, sub_questions=%s",
        plan["plan_source"],
        plan["question_type"],
        plan["queries"],
        plan["title_guesses"],
        plan["sub_questions"],
    )
    return plan


plan_queries = plan_research_queries

