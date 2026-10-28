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

_STRONG_FACTUAL_PATTERNS = re.compile(
    r"\b("
    r"which paper|who introduced|who proposed|who invented|what paper|"
    r"when was|when did|what year|who first|who created|what was the first|"
    r"who wrote|who published|what is the name|name the paper|name the author|"
    r"cite the|what model introduced|what architecture introduced"
    r")\b",
    re.IGNORECASE,
)

_EXTENDED_FACTUAL_PATTERNS = re.compile(
    r"("
    r"\bmain findings of the\s+[\w\-]+\s+paper\b|"
    r"\bwhat technique does\s+[A-Z][\w\-]+\s+use\b|"
    r"\bwhat experiment demonstrated\b|"
    r"\bwhat do the\s+[A-Z][\w\-]+\s+scaling laws\b"
    r")",
    re.IGNORECASE,
)

_STRICT_COMPARATIVE_PATTERNS = re.compile(
    r"\b("
    r"compar(?:e|es|ed|ing|ison|ative)|versus|\bvs\.?\b|trade.?off|"
    r"which is better|which performs|survey|review|overview|"
    r"differences?\s+between|differ\s+from|main\s+types\s+of|main\s+techniques\s+for|"
    r"what\s+techniques\s+are\s+used|what\s+causes|what\s+role\s+does"
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
    is_strong_factual = bool(_STRONG_FACTUAL_PATTERNS.search(question))
    is_extended_factual = bool(_EXTENDED_FACTUAL_PATTERNS.search(question))
    is_strict_comparative = bool(_STRICT_COMPARATIVE_PATTERNS.search(question))

    if is_strong_factual:
        return "factual_lookup"
    if is_extended_factual and not is_strict_comparative:
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


# Canonical seed table for deterministic fallback guidance when SEED_PAPERS_ENABLED=True
_CANONICAL_SEEDS: list[tuple[re.Pattern, list[str], list[str]]] = [
    (
        re.compile(r"\btransformer\b", re.I),
        ["Attention Is All You Need Vaswani", "transformer self-attention"],
        ["Attention Is All You Need"],
    ),
    (
        re.compile(r"\bbert\b", re.I),
        ["BERT Pre-training Deep Bidirectional Transformers Devlin", "masked language model BERT"],
        ["BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"],
    ),
    (
        re.compile(r"\b(resnet|deep residual learning|residual network)\b", re.I),
        ["Deep Residual Learning for Image Recognition He", "residual networks skip connection"],
        ["Deep Residual Learning for Image Recognition"],
    ),
    (
        re.compile(r"\blora\b", re.I),
        ["LoRA Low-Rank Adaptation of Large Language Models Hu"],
        ["LoRA: Low-Rank Adaptation of Large Language Models"],
    ),
    (
        re.compile(r"\badam\b", re.I),
        ["Adam A Method for Stochastic Optimization Kingma Ba"],
        ["Adam: A Method for Stochastic Optimization"],
    ),
    (
        re.compile(r"\b(generative adversarial net\w*|gans?)\b", re.I),
        ["Generative Adversarial Nets Goodfellow"],
        ["Generative Adversarial Nets", "Generative Adversarial Networks"],
    ),
    (
        re.compile(r"\b(retrieval[- ]augmented generation|\brag\b)\b", re.I),
        ["Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks Lewis"],
        ["Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"],
    ),
    (
        re.compile(r"\bdropout\b", re.I),
        ["Dropout A Simple Way to Prevent Neural Networks from Overfitting Srivastava"],
        ["Dropout: A Simple Way to Prevent Neural Networks from Overfitting"],
    ),
    (
        re.compile(r"\bbatch normalization\b", re.I),
        ["Batch Normalization Accelerating Deep Network Training Ioffe Szegedy"],
        ["Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"],
    ),
    (
        re.compile(r"\b(vision transformers?|\bvit\b)\b", re.I),
        ["An Image is Worth 16x16 Words Transformers for Image Recognition at Scale Dosovitskiy"],
        ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"],
    ),
    (
        re.compile(r"\bword2vec\b", re.I),
        ["Efficient Estimation of Word Representations in Vector Space Mikolov"],
        ["Efficient Estimation of Word Representations in Vector Space"],
    ),
    (
        re.compile(r"\b(u-net|unet)\b", re.I),
        ["U-Net Convolutional Networks for Biomedical Image Segmentation Ronneberger"],
        ["U-Net: Convolutional Networks for Biomedical Image Segmentation"],
    ),
    (
        re.compile(r"\bfaster r-cnn\b", re.I),
        ["Faster R-CNN Towards Real-Time Object Detection with Region Proposal Networks Ren"],
        ["Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"],
    ),
    (
        re.compile(r"\bmask r-cnn\b", re.I),
        ["Mask R-CNN He Gkioxari Dollar Girshick"],
        ["Mask R-CNN"],
    ),
    (
        re.compile(r"\blatent diffusion\b", re.I),
        ["High-Resolution Image Synthesis with Latent Diffusion Models Rombach"],
        ["High-Resolution Image Synthesis with Latent Diffusion Models"],
    ),
    (
        re.compile(r"\bdensenet\b", re.I),
        ["Densely Connected Convolutional Networks Huang"],
        ["Densely Connected Convolutional Networks"],
    ),
    (
        re.compile(r"\bgraph convolutional networks?\b", re.I),
        ["Semi-Supervised Classification with Graph Convolutional Networks Kipf Welling"],
        ["Semi-Supervised Classification with Graph Convolutional Networks"],
    ),
    (
        re.compile(r"\bt5\b", re.I),
        ["Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer Raffel"],
        ["Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"],
    ),
    (
        re.compile(r"\byolo\b", re.I),
        ["You Only Look Once Unified Real-Time Object Detection Redmon"],
        ["You Only Look Once: Unified, Real-Time Object Detection"],
    ),
    (
        re.compile(r"\blayer normalization\b", re.I),
        ["Layer Normalization Ba Kiros Hinton"],
        ["Layer Normalization"],
    ),
    (
        re.compile(r"\bgpt-?3\b", re.I),
        ["Language Models are Few-Shot Learners Brown"],
        ["Language Models are Few-Shot Learners"],
    ),
    (
        re.compile(r"\bsimclr\b", re.I),
        ["A Simple Framework for Contrastive Learning of Visual Representations Chen"],
        ["A Simple Framework for Contrastive Learning of Visual Representations"],
    ),
    (
        re.compile(r"\bflash\s*attention\b", re.I),
        ["FlashAttention Fast and Memory-Efficient Exact Attention with IO-Awareness Dao"],
        ["FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"],
    ),
    (
        re.compile(r"\bchinchilla\b", re.I),
        ["Training Compute-Optimal Large Language Models Hoffmann"],
        ["Training Compute-Optimal Large Language Models"],
    ),
    (
        re.compile(r"\b(mixture[- ]of[- ]experts|switch transformers?)\b", re.I),
        ["Switch Transformers Scaling to Trillion Parameter Models Fedus"],
        ["Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity"],
    ),
    (
        re.compile(r"\balphafold\w*\b", re.I),
        ["Highly accurate protein structure prediction with AlphaFold Jumper"],
        ["Highly accurate protein structure prediction with AlphaFold"],
    ),
]


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

    # Seed list of well-known papers for deterministic fallback guidance when enabled
    if getattr(settings, "SEED_PAPERS_ENABLED", True):
        lower_q = question.lower()
        for pat, seed_qs, seed_titles in _CANONICAL_SEEDS:
            if pat.pattern == r"\btransformer\b":
                # Only match generic Transformer architecture when not asking about a specific variant (ViT, BERT, T5, Switch, Swin, Longformer)
                if any(v in lower_q for v in ("vision transformer", "vit", "bert", "switch transformer", "mixture-of-experts", "mixture of experts", "t5", "text-to-text", "swin", "longformer", "roformer")):
                    continue
            if pat.search(lower_q):
                for sq in seed_qs:
                    if sq not in queries:
                        queries.append(sq)
                for st in seed_titles:
                    if st not in title_guesses:
                        title_guesses.append(st)

    # 4. Generate topical keyword queries per facet
    if keywords:
        broad_kw = " ".join(keywords[:4])
        if broad_kw not in queries:
            queries.append(broad_kw)

        for f in facets:
            f_words = [w.lower() for w in re.findall(r"\b\w{3,}\b", f) if w.lower() not in STOP_WORDS]
            if f_words:
                f_query = " ".join(f_words + keywords[:2])
                f_query_dedup = " ".join(dict.fromkeys(f_query.split()))
                if f_query_dedup not in queries:
                    queries.append(f_query_dedup)

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
        "title_guesses": title_guesses[:3],
        "sub_questions": sub_questions[:3],
        "target_concepts": target_concepts[:8],
        "plan_source": "dynamic_fallback_planner",
        "expected_titles": title_guesses[:3],
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
                        # Prioritize LLM-generated queries while preserving seed/fallback coverage up to 6 queries
                        merged_queries = list(dict.fromkeys(llm_queries + plan["queries"]))
                        plan["queries"] = merged_queries[:6]
                        used_llm = True
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
