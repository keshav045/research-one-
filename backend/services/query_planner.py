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
from typing import TypedDict

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


class QueryPlan(TypedDict):
    question_type: str
    queries: list[str]
    title_guesses: list[str]
    sub_questions: list[str]


def fallback_query_planner(question: str) -> QueryPlan:
    """
    Deterministic rule-based query generator when LLM is unavailable or fails.
    Removes stop words and meta words, extracts named paper/model concepts.
    """
    q_type = classify_question_type(question)

    # 1. Clean tokens
    tokens = re.sub(r"[^a-zA-Z0-9\s-]", " ", question.lower()).split()
    meaningful = [t for t in tokens if len(t) > 2 and t not in STOP_WORDS]
    keywords = [t for t in meaningful if t not in META_WORDS]

    queries: list[str] = []
    title_guesses: list[str] = []
    # Default sub-question fallback: use the original question as the only sub-question
    sub_questions = [question.strip()]

    # Seed list of well-known papers for deterministic fallback guidance;
    # not the planner's main logic. Disabled when SEED_PAPERS_ENABLED=False.
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

    # Add generic cleaned keyword query
    if keywords:
        clean_kw = " ".join(keywords[:5])
        if clean_kw not in queries:
            queries.append(clean_kw)

    # Add secondary sub-phrase if long
    if len(keywords) > 3:
        sub_phrase = " ".join(keywords[:3])
        if sub_phrase not in queries:
            queries.append(sub_phrase)

    return {
        "question_type": q_type,
        "queries": queries[:4],
        "title_guesses": title_guesses[:2],
        "sub_questions": sub_questions,
        "plan_source": "fallback",
    }


async def plan_research_queries(question: str) -> QueryPlan:
    """
    Plan search queries and sub-questions using LLM, with fallback to original question as only sub-question.
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
            logger.warning("[QueryPlanner] Fallback plan used: LLM returned empty reply")
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

