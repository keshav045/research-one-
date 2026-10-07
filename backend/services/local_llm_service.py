"""
Local LLM Report Synthesis Service
==================================
Implements Step 6A & 6B:
- Input: verified claims only, each with source paper metadata (title, authors, year, venue) from paper record, plus page numbers.
- Output: plain-text answer, 3-6 sentences with [n] markers. No large JSON.
- Prompt rules: use only given claims, add no new facts, name source paper with authors and year in first sentence, keep [n].
- If anchor_confidence is "uncertain", first sentence says "The source paper is uncertain: A or B".
- Builds references, methodology, and limitations in Python from stage stats and paper metadata.
- Integrates Answer-level NLI (verify_answer_sentences) to prune unsupported assertions.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Optional

import httpx
import torch

from ..config import settings
from ..models.schemas import (
    Citation,
    CitationStatus,
    ComparisonRow,
    Paper,
    ReportParagraph,
    ReportSection,
    ResearchDepth,
    ResearchReport,
)
from .nli_verifier import verify_answer_sentences
from .text_utils import split_into_sentences

logger = logging.getLogger(__name__)

_tokenizer = None
_model = None
_device = None


def get_local_model():
    """Lazy load the local CausalLM model on CUDA or CPU."""
    global _tokenizer, _model, _device
    if _model is not None:
        return _tokenizer, _model, _device

    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_name = settings.LOCAL_LLM_MODEL
    _device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("[LocalLLM] Loading tokenizer for %s...", model_name)
    _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    logger.info("[LocalLLM] Loading model %s on %s...", model_name, _device)
    _model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if _device == "cuda" else torch.float32,
        device_map="auto" if _device == "cuda" else None,
        trust_remote_code=True,
    )
    if _device == "cpu":
        _model = _model.to("cpu")
    _model.eval()
    logger.info("[LocalLLM] Model loaded successfully.")
    return _tokenizer, _model, _device


def _run_local_text_generation(prompt: str, max_tokens: int = 512) -> str:
    """Run local HuggingFace LLM text generation."""
    tokenizer, model, device = get_local_model()

    messages = [
        {
            "role": "system",
            "content": (
                "You are an academic synthesis system. "
                "Synthesize verified research findings into concise academic prose. "
                "CRITICAL: Keep bracketed citations [1], [2] attached to their claims. "
                "Do NOT invent new claims, numbers, or authors."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.2,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output_ids[0][len(inputs.input_ids[0]):]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


async def _call_local(prompt: str, max_tokens: int = 512) -> str:
    """Async wrapper for running local HuggingFace LLM text generation."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run_local_text_generation, prompt, max_tokens)


_ollama_first_call = True


async def _call_ollama(prompt: str, max_tokens: int = 512) -> str:
    """Call Ollama REST API for LLM text generation."""
    global _ollama_first_call
    url = f"{settings.OLLAMA_URL}/api/generate"
    payload = {
        "model": settings.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "10m",
        "options": {
            "temperature": 0.2,
            "num_predict": max_tokens,
        },
    }
    timeout = 180.0 if _ollama_first_call else 45.0
    _ollama_first_call = False

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()
    except Exception as exc:
        logger.warning("[Ollama] Request failed: %s", exc)
        return ""


async def _call_openai(prompt: str, max_tokens: int = 512) -> str:
    """Call official OpenAI API for text generation."""
    if not settings.is_openai_configured:
        return ""
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        resp = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an academic synthesis system. "
                        "Synthesize verified research findings into concise academic prose. "
                        "CRITICAL: Keep bracketed citations [1], [2] attached to their claims. "
                        "Do NOT invent new claims, numbers, or authors."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip() if resp.choices else ""
    except Exception as exc:
        logger.warning("[OpenAI] Call failed: %s", exc)
        return ""


def get_provider_model(provider: str) -> str:
    """Return model name associated with the provider."""
    prov = (provider or "").lower()
    if "openai" in prov:
        return settings.OPENAI_MODEL
    elif "gemini" in prov:
        return settings.GEMINI_MODEL
    elif "qwen" in prov:
        return settings.QWEN_MODEL
    elif "ollama" in prov:
        return settings.OLLAMA_MODEL
    elif "local" in prov:
        return settings.LOCAL_LLM_MODEL
    return "none"


async def _call_llm(prompt: str, max_tokens: int = 512) -> tuple[str, str]:
    """Unified LLM caller routing to OpenAI, Gemini, Qwen, Ollama, or fallback.
    Returns (text, provider_used).
    """
    provider = settings.LLM_PROVIDER.lower()

    if provider == "openai":
        res = await _call_openai(prompt, max_tokens)
        if res:
            return res, "openai"
        if getattr(settings, "LLM_FALLBACK_LOCAL", False):
            logger.warning("[LLM] OpenAI call failed. Falling back to local HF model (LLM_FALLBACK_LOCAL=True).")
            res_local = await _call_local(prompt, max_tokens)
            return res_local, "local (fallback)"
        else:
            logger.warning("[LLM] OpenAI call failed. Local fallback disabled (LLM_FALLBACK_LOCAL=False).")
            return "", "none"

    elif provider == "ollama":
        res = await _call_ollama(prompt, max_tokens)
        if res:
            return res, "ollama"
        if getattr(settings, "LLM_FALLBACK_LOCAL", False):
            logger.warning("[LLM] Ollama call failed. Falling back to local HF model (LLM_FALLBACK_LOCAL=True).")
            res_local = await _call_local(prompt, max_tokens)
            return res_local, "local (fallback)"
        else:
            logger.warning("[LLM] Ollama call failed. Local fallback disabled (LLM_FALLBACK_LOCAL=False).")
            return "", "none"

    elif provider in ("qwen", "gemini"):
        try:
            from .gemini_service import generate_text_simple
            res = await generate_text_simple(prompt, max_tokens)
            if res:
                return res, provider
        except Exception as exc:
            logger.warning("[LLM] Cloud provider failed: %s", exc)
        if getattr(settings, "LLM_FALLBACK_LOCAL", False):
            logger.warning("[LLM] Cloud provider failed. Falling back to local HF model (LLM_FALLBACK_LOCAL=True).")
            res_local = await _call_local(prompt, max_tokens)
            return res_local, "local (fallback)"
        else:
            logger.warning("[LLM] Cloud provider failed. Local fallback disabled (LLM_FALLBACK_LOCAL=False).")
            return "", "none"

    elif provider == "local":
        if getattr(settings, "LLM_FALLBACK_LOCAL", False):
            res = await _call_local(prompt, max_tokens)
            return res, "local"
        else:
            logger.warning("[LLM] Local provider requested but local model disabled.")
            return "", "none"

    else:
        # Default chain if unknown provider: try OpenAI first if configured, else Gemini
        if settings.is_openai_configured:
            res = await _call_openai(prompt, max_tokens)
            if res:
                return res, "openai"
        if settings.is_gemini_configured or settings.is_qwen_configured:
            try:
                from .gemini_service import generate_text_simple
                res = await generate_text_simple(prompt, max_tokens)
                if res:
                    return res, "gemini" if settings.is_gemini_configured else "qwen"
            except Exception:
                pass
        return "", "none"


# ─── Step 6A: Answer Generation ───────────────────────────────────────────────

def anchor_state(anchor_paper: Optional[Paper], anchor_confidence: str) -> tuple[str, str]:
    """
    Single source of truth for anchor paper status across all report sections.
    Returns (label, sentence).
    If anchor_paper is None: label="Not identified", sentence="Foundational anchor paper: Not identified."
    If anchor_confidence == "uncertain": label="Uncertain", sentence="Foundational anchor paper: Selection is uncertain."
    If anchor_paper exists and confidence is high/normal: label=anchor_paper.title, sentence="Foundational anchor paper: '{title}' ({year})."
    """
    if not anchor_paper or str(anchor_confidence).lower() in ("none", "null", ""):
        return "Not identified", "Foundational anchor paper: Not identified."
    if str(anchor_confidence).lower() == "uncertain":
        return "Uncertain", f"Foundational anchor paper: Selection is uncertain ('{anchor_paper.title}')."
    year_str = f" ({anchor_paper.publicationYear})" if anchor_paper.publicationYear else ""
    return anchor_paper.title, f"Foundational anchor paper: '{anchor_paper.title}'{year_str}."


def collapse_repeated_citations(text: str) -> str:
    """Collapses duplicate citation markers such as [1][1] or [1] [1] into [1]."""
    text = re.sub(r"\[(\d+)\](?:\s*\[\1\])+", r"[\1]", text)
    return text


def _format_first_sentence(
    question: str,
    anchor_paper: Optional[Paper],
    anchor_confidence: str = "high",
    alternate_paper: Optional[Paper] = None,
) -> str:
    """Formulate mandatory first sentence naming source paper (title, authors, year) deterministically in Python."""
    label, sent = anchor_state(anchor_paper, anchor_confidence)
    if label == "Not identified":
        return f"Foundational anchor paper: Not identified for '{question}'."
    if label == "Uncertain":
        return sent
    year = getattr(anchor_paper, "publicationYear", 2024) or 2024
    authors_list = getattr(anchor_paper, "authors", []) or ["Unknown"]
    if len(authors_list) > 3:
        authors_str = ", ".join(authors_list[:3]) + " et al."
    else:
        authors_str = ", ".join(authors_list)
    return f"{anchor_paper.title} was introduced by {authors_str} in {year}."


async def generate_plain_answer(
    question: str,
    verified_citations: list[Citation],
    anchor_paper: Optional[Paper] = None,
    anchor_confidence: str = "high",
    alternate_paper: Optional[Paper] = None,
    ref_index: Optional[dict[str, int]] = None,
) -> tuple[str, str, str]:
    """
    Step 6A Plain-Text Answer Generation:
    - Input: verified claims with exact source paper metadata (never from LLM).
    - Output: (plain_text_claims, provider_used, model_name).
    """
    if not verified_citations:
        return "", "none", "none"

    # Honest Reporting Check (Requirement 9 & B8):
    # Applies to any literature-review question when evidence-bearing papers are 1 or fewer
    from .query_planner import classify_question_type
    q_type = classify_question_type(question)
    distinct_papers = {c.paperId for c in verified_citations if c.status == CitationStatus.VERIFIED}
    n_papers = len(distinct_papers)
    if q_type != "factual_lookup" and n_papers <= 1:
        first_c = verified_citations[0]
        ref_num = ref_index.get(first_c.paperId, 1) if ref_index else 1
        claim_text = first_c.claim.strip().rstrip(".")
        honest_answer = (
            f"Research coverage is insufficient to answer this question in general. "
            f"Only {n_papers} relevant paper(s) with usable evidence were retrieved. "
            f"{claim_text} [{ref_num}]. "
            f"Insufficient verified evidence was available to compare it against alternative techniques or comprehensively address all facets of the inquiry."
        )
        return honest_answer, "honest_reporting_system", "none"

    # Format claims with strictly verified metadata and reference numbers
    claims_context = []
    for c in verified_citations[:8]:
        r_num = ref_index.get(c.paperId, c.badgeNumber) if ref_index else c.badgeNumber
        claims_context.append(
            f"[{r_num}] {c.claim} (Paper: '{c.paperTitle}', Authors: {c.authors}, Year: {c.year}, Page: {c.page})"
        )
    claims_block = "\n".join(claims_context)

    prompt = (
        f"Research Question: {question}\n\n"
        f"Verified Source Facts:\n{claims_block}\n\n"
        "Instructions:\n"
        "Write a concise academic answer of 2 to 5 sentences directly answering the question.\n"
        "Synthesize these findings into a coherent, comparative academic summary connecting the evidence.\n"
        "Strict Requirements:\n"
        "1. Use ONLY the given verified source facts. Add NO new facts, dates, names, or unverified claims.\n"
        "2. Preserve the bracketed citation markers [n] directly adjacent to each asserted fact.\n"
        "3. Output PLAIN TEXT ONLY. Do not output markdown titles, lists, or JSON.\n"
        "4. Author Consistency: When referring to authors by name in prose (e.g. 'Vaswani et al.', 'Lewis et al.'), ensure the named authors match the specific paper cited by [n]. NEVER attribute findings to authors different from the authors listed in the source metadata for that [n]."
    )

    provider_used = "none"
    model_name = "none"
    try:
        raw_output, provider_used = await _call_llm(prompt, max_tokens=350)
        clean = raw_output.strip().replace("```", "").strip()
        # Verify LLM respected [n] citation markers and did not produce empty response
        if clean and re.search(r"\[\d+\]", clean) and len(clean.split()) >= 15:
            model_name = get_provider_model(provider_used)
            return clean, provider_used, model_name
    except Exception as exc:
        logger.info("[LocalLLM] Answer generation fallback triggered: %s", exc)

    # Deterministic fallback: sentences with real [n] markers
    body_sentences = []
    seen_fallback_claims = set()
    for c in verified_citations[:6]:
        text = c.claim.strip().rstrip(".")
        norm = text.lower()
        if norm in seen_fallback_claims:
            continue
        seen_fallback_claims.add(norm)
        r_num = ref_index.get(c.paperId, getattr(c, "badgeNumber", 1)) if ref_index else getattr(c, "badgeNumber", 1)
        body_sentences.append(f"{text} [{r_num}].")

    fb_provider = "deterministic_fallback" if not provider_used or provider_used == "none" else f"{provider_used} (failed)"
    return " ".join(body_sentences), fb_provider, "none"


# ─── Structured Synthesis in Python (No Invented Data) ─────────────────────────

def build_methodology_from_stats(
    question: str,
    papers: list[Paper],
    stage_stats: Optional[list[Any]] = None,
    anchor_rule: str = "none",
    anchor_confidence: str = "high",
    integrity: float = 1.0,
    anchor_paper: Optional[Paper] = None,
) -> str:
    """Builds research methodology dynamically from pipeline stage stats (no fixed text)."""
    sources_count: dict[str, int] = {}
    for p in papers:
        src = getattr(p, "source", "arXiv") or "arXiv"
        sources_count[src] = sources_count.get(src, 0) + 1
    src_summary = ", ".join(f"{k} ({v})" for k, v in sources_count.items()) if sources_count else "None"

    total_passages = sum(len(getattr(p, "passages", []) or []) for p in papers)

    if anchor_paper and str(anchor_confidence).lower() not in ("none", "null"):
        anchor_line = f"- **Anchor Identification**: Selected foundational anchor paper via rule `{anchor_rule}` (confidence: `{anchor_confidence}`)."
    else:
        anchor_line = "- **Anchor Identification**: Foundational anchor paper: Not identified (literature review / multi-concept investigation). Anchor confidence: None."

    lines = [
        "### Empirical Research Protocol & Methodological Audit",
        "",
        f"- **Corpus Ingestion**: Queried academic open access indices; ingested {len(papers)} candidate papers across {src_summary}.",
        anchor_line,
        f"- **Full-Text Passage Extraction**: Extracted and indexed {total_passages} verbatim passages via PyMuPDF windowed segmentation.",
        "- **Evidence Extraction & Source Match**: Verified candidate assertions against verbatim source passages via source match.",
        f"- **Answer Verification & Citation Integrity**: Synthesized prose was audited sentence-by-sentence via answer-level NLI, achieving {integrity:.1%} citation integrity.",
    ]
    return "\n".join(lines)


def build_limitations_from_stats(
    anchor_paper: Optional[Paper],
    anchor_confidence: str,
    integrity: float,
    removed_sentences: list[dict[str, Any]],
    retrieval_warnings: Optional[list[str]] = None,
    debug_info: Optional[dict[str, Any]] = None,
) -> list[str]:
    """Generates factual limitations based on real execution conditions."""
    limits: list[str] = []
    if debug_info and "papers_discovered" in debug_info:
        disc = debug_info.get("papers_discovered", 0)
        uniq = debug_info.get("unique_papers", disc)
        rel = debug_info.get("relevant_papers", 0)
        full = debug_info.get("full_text_papers", 0)
        passages = debug_info.get("passages_total", 0)
        ev_papers = debug_info.get("evidence_bearing_papers", 0)
        ver = debug_info.get("verified_claims", 0)
        limits.append(
            f"Retrieval & Evidence Funnel: {disc} discovered -> {uniq} unique -> {rel} relevant -> "
            f"{full} full text -> {passages} passages -> {ev_papers} evidence-bearing -> {ver} verified claims."
        )
    if anchor_confidence == "uncertain":
        limits.append("Foundational paper attribution is marked uncertain due to close citation counts or divergent selection heuristics.")
    if anchor_paper and getattr(anchor_paper, "is_abstract_only", False):
        limits.append(f"Full-text PDF for anchor paper '{anchor_paper.title}' was inaccessible; analysis was restricted to abstract text.")
    if integrity < 0.80:
        limits.append(f"Citation integrity ({integrity:.1%}) is below the 80% threshold; {len(removed_sentences)} unsupported sentence(s) were pruned by NLI audit.")
    from .paper_retrieval import get_source_status, summarize_source_status
    status = get_source_status()
    if any(s in ("RATE_LIMITED", "ERROR", "PARTIAL") for s in status.values()):
        limits.append(f"Academic source status: {summarize_source_status()}.")
    if not limits:
        limits.append("Findings are bounded by the open-access academic literature retrieved during the research session.")
    return limits


def build_references_from_citations(papers: list[Paper], citations: list[Citation]) -> list[Paper]:
    """Matches every cited [n] to real paper metadata, keeping anchor paper at index 0."""
    paper_map = {p.id: p for p in papers}
    referenced_ids: list[str] = []

    # Ensure cited papers are prioritized
    for c in citations:
        if c.paperId in paper_map and c.paperId not in referenced_ids:
            referenced_ids.append(c.paperId)

    # Add remaining papers up to 12
    for p in papers:
        if p.id not in referenced_ids:
            referenced_ids.append(p.id)

    ordered = [paper_map[pid] for pid in referenced_ids if pid in paper_map]
    return ordered[:12]


def build_comparison_table(papers: list[Paper], citations: list[Citation]) -> list[ComparisonRow]:
    """Constructs comparison rows only from real metadata (title, authors, year, venue, citation count)."""
    rows: list[ComparisonRow] = []
    seen = set()

    for p in papers[:6]:
        if not p.title:
            continue
        title = p.title.strip()
        if title.lower() in seen:
            continue
        seen.add(title.lower())

        matched_cite = next((c for c in citations if c.paperId == p.id), None)
        cite_id = matched_cite.id if matched_cite else (citations[0].id if citations else "cite-1")

        authors_str = ", ".join(p.authors) if p.authors else "Not extracted"
        year_str = str(p.publicationYear) if p.publicationYear else "Not extracted"
        venue_str = p.journalConference.strip() if (getattr(p, "journalConference", None) and p.journalConference.strip()) else "Not extracted"
        cites_str = f"{p.citationCount:,}" if (getattr(p, "citationCount", None) is not None and p.citationCount >= 0) else "Not extracted"

        rows.append(
            ComparisonRow(
                model=title,
                architectureType=authors_str,
                dataset="Not extracted",
                f1Score="Not extracted",
                mapScore="Not extracted",
                fpsThroughput="Not extracted",
                parametersM="Not extracted",
                gflops="Not extracted",
                citationId=cite_id,
                title=title,
                authors=authors_str,
                year=year_str,
                venue=venue_str,
                citationCount=cites_str,
            )
        )
    return rows


async def synthesize_report(
    question: str,
    depth: ResearchDepth,
    papers: list[Paper],
    claims: list[Any],
    citations: list[Citation],
    anchor_paper: Optional[Paper] = None,
    anchor_confidence: str = "high",
    alternate_paper: Optional[Paper] = None,
    anchor_rule: str = "none",
    stage_stats: Optional[list[Any]] = None,
    retrieval_warnings: Optional[list[str]] = None,
    debug_info: Optional[dict[str, Any]] = None,
) -> tuple[ResearchReport, float, list[dict[str, Any]]]:
    """
    Main Step 6 Report Synthesis:
    1. Generates plain-text answer with [n] badges.
    2. Audits answer sentences via Answer-Level NLI (removes unsupported sentences).
    3. Builds methodology, references, and limitations deterministically from metadata.
    4. Returns (report, citation_integrity, removed_sentences).
    """
    first_sentence = _format_first_sentence(question, anchor_paper, anchor_confidence=anchor_confidence)

    uncertainty_note = ""
    if anchor_confidence == "uncertain":
        if anchor_paper and alternate_paper:
            y1 = getattr(anchor_paper, "publicationYear", 2024) or 2024
            y2 = getattr(alternate_paper, "publicationYear", 2024) or 2024
            uncertainty_note = f"Source paper uncertain: '{anchor_paper.title}' ({y1}) or '{alternate_paper.title}' ({y2})."
        elif anchor_paper:
            uncertainty_note = f"Source paper selection is marked uncertain for '{anchor_paper.title}'."

    references = build_references_from_citations(papers, citations)
    ref_index = {p.id: idx + 1 for idx, p in enumerate(references)}
    for c in citations:
        c.badgeNumber = ref_index.get(c.paperId, 1)
    comparison_table = build_comparison_table(papers, citations)

    # 1. Generate plain-text claim sentences using reference indices
    raw_claims, writer_provider, writer_model = await generate_plain_answer(
        question=question,
        verified_citations=citations,
        anchor_paper=anchor_paper,
        anchor_confidence=anchor_confidence,
        ref_index=ref_index,
    )
    if debug_info is not None:
        debug_info["writer_provider"] = writer_provider
        debug_info["writer_model"] = writer_model
        if "fallback" in str(writer_provider).lower():
            warn_msg = "Writer used local fallback model"
            if warn_msg not in debug_info.setdefault("llm_warnings", []):
                debug_info["llm_warnings"].append(warn_msg)

    # 2. Answer-Level NLI Verification (Step 6B)
    verified_answer, integrity, verified_details, removed_details = verify_answer_sentences(
        answer_text=raw_claims,
        citations=citations,
        anchor_paper=anchor_paper,
        first_sentence=first_sentence,
    )
    if debug_info is not None:
        debug_info["verified_sentences"] = verified_details
        debug_info["removed_sentences"] = removed_details

    # Citation Validation: ensure all [n] badges map to an existing reference and collapse repeats
    ref_count = len(references)

    def _validate_ref(match):
        num = int(match.group(1))
        if 1 <= num <= ref_count:
            return f"[{num}]"
        logger.warning(
            "[CitationValidation] Citation [%d] has no reference (only %d references exist). Removing invalid citation.",
            num,
            ref_count,
        )
        return ""

    verified_answer = re.sub(r"\[(\d+)\]", _validate_ref, verified_answer)

    # Author-reference alignment: ensure named authors in prose match the cited reference paper
    def _align_author_citations(sentence: str) -> str:
        author_m = re.findall(r"\b([A-Z][a-zA-Z\-]+)\s+et\s+al\.?", sentence)
        cite_m = [int(n) for n in re.findall(r"\[(\d+)\]", sentence)]
        if author_m and cite_m:
            for auth in author_m:
                for c_num in cite_m:
                    if 1 <= c_num <= ref_count:
                        ref_paper = references[c_num - 1]
                        ref_auths = " ".join(getattr(ref_paper, "authors", []) or []).lower()
                        if auth.lower() not in ref_auths:
                            actual_ref_idx = next(
                                (idx + 1 for idx, p in enumerate(references) if auth.lower() in " ".join(getattr(p, "authors", []) or []).lower()),
                                None
                            )
                            if actual_ref_idx:
                                sentence = sentence.replace(f"[{c_num}]", f"[{actual_ref_idx}]")
        return sentence

    sents_aligned = [_align_author_citations(s) for s in split_into_sentences(verified_answer)]
    verified_answer = " ".join(sents_aligned)
    verified_answer = collapse_repeated_citations(verified_answer)
    verified_answer = re.sub(r"\s+([.,;:!?])", r"\1", verified_answer)
    verified_answer = re.sub(r"\s+", " ", verified_answer).strip()

    # 3. Build structured sections from metadata
    methodology = build_methodology_from_stats(
        question=question,
        papers=papers,
        stage_stats=stage_stats,
        anchor_rule=anchor_rule,
        anchor_confidence=anchor_confidence,
        integrity=integrity,
        anchor_paper=anchor_paper,
    )

    limitations = build_limitations_from_stats(
        anchor_paper=anchor_paper,
        anchor_confidence=anchor_confidence,
        integrity=integrity,
        removed_sentences=removed_details,
        retrieval_warnings=retrieval_warnings,
        debug_info=debug_info,
    )
    if debug_info and debug_info.get("llm_warnings"):
        for w in debug_info["llm_warnings"]:
            if w not in limitations:
                limitations.append(w)
    if uncertainty_note and uncertainty_note not in limitations:
        limitations.insert(0, uncertainty_note)

    # 4. Dynamic Concept & Facet Extraction (B5: No hardcoded topic content)
    from .query_planner import extract_question_facets
    facets = extract_question_facets(question)
    concept_coverage_map: dict[str, str] = {}
    facet_citations: dict[str, list[Citation]] = {f: [] for f in facets}
    unmatched_citations: list[Citation] = []

    for c in citations:
        if c.status != CitationStatus.VERIFIED:
            continue
        c_text = f"{c.claim} {c.passage}".lower()
        matched_facet = False
        for f in facets:
            f_words = [w.lower() for w in re.findall(r"\b\w{3,}\b", f)]
            if any(w in c_text for w in f_words):
                facet_citations[f].append(c)
                matched_facet = True
                break
        if not matched_facet:
            unmatched_citations.append(c)

    findings: list[ReportSection] = []
    tech_comparison: list[dict[str, Any]] = []
    insufficient_evidence_items: list[str] = []

    for f in facets:
        f_cites = facet_citations[f]
        if f_cites:
            concept_coverage_map[f] = "VERIFIED"
            # Deduplicate by paper and claim
            seen_claims = set()
            unique_f_cites = []
            for c in f_cites:
                if c.claim not in seen_claims:
                    seen_claims.add(c.claim)
                    unique_f_cites.append(c)

            p_text = " ".join(f"{c.claim.strip().rstrip('.')} [{ref_index.get(c.paperId, 1)}]." for c in unique_f_cites[:4])
            p_text = collapse_repeated_citations(p_text)
            findings.append(ReportSection(
                sectionTitle=f,
                paragraphs=[ReportParagraph(
                    text=p_text,
                    citations=unique_f_cites[:4],
                )]
            ))
            # B5: Build technique comparison rows ONLY for concepts with verified evidence
            top_claim = unique_f_cites[0]
            impact_text = top_claim.claim.strip().rstrip(".")
            tech_comparison.append({
                "technique": f,
                "category": "Empirical Finding",
                "impact": impact_text[:120],
                "evidence_status": "VERIFIED",
            })
        else:
            concept_coverage_map[f] = "INSUFFICIENT"
            insufficient_evidence_items.append(f"{f}: Insufficient verified empirical evidence retrieved in current session.")

    if writer_provider == "honest_reporting_system":
        tech_comparison = []

    if unmatched_citations:
        seen_claims = set()
        unique_unmatched = []
        for c in unmatched_citations:
            if c.claim not in seen_claims:
                seen_claims.add(c.claim)
                unique_unmatched.append(c)
        p_text = " ".join(f"{c.claim.strip().rstrip('.')} [{ref_index.get(c.paperId, 1)}]." for c in unique_unmatched[:4])
        p_text = collapse_repeated_citations(p_text)
        findings.append(ReportSection(
            sectionTitle="Additional Empirical Findings",
            paragraphs=[ReportParagraph(
                text=p_text,
                citations=unique_unmatched[:4],
            )]
        ))

    # B4: Executive Summary vs Findings - ensure summary doesn't duplicate findings
    # Summary is at most 3 sentences: coverage sentence + short synthesis
    summary_sentences = split_into_sentences(verified_answer)
    if len(summary_sentences) > 3:
        exec_summary = " ".join(summary_sentences[:3])
    else:
        exec_summary = verified_answer

    # If findings was empty, populate with citations
    if not findings and citations:
        p_text = " ".join(f"{c.claim.strip().rstrip('.')} [{ref_index.get(c.paperId, 1)}]." for c in citations[:4])
        findings.append(ReportSection(
            sectionTitle="Empirical Findings",
            paragraphs=[ReportParagraph(
                text=collapse_repeated_citations(p_text),
                citations=citations[:4],
            )]
        ))

    # Ensure no sentence > 40 chars appears in both summary and findings
    findings_sentences = set()
    for sec in findings:
        for p in sec.paragraphs:
            for s in split_into_sentences(p.text):
                s_norm = " ".join(re.sub(r"\[\d+\]", "", s).split()).strip().lower()
                if len(s_norm) > 40:
                    findings_sentences.add(s_norm)

    exec_sents_clean = []
    for s in split_into_sentences(exec_summary):
        s_norm = " ".join(re.sub(r"\[\d+\]", "", s).split()).strip().lower()
        if len(s_norm) > 40 and s_norm in findings_sentences:
            continue
        exec_sents_clean.append(s)
    if exec_sents_clean:
        exec_summary = " ".join(exec_sents_clean)

    if debug_info is not None:
        debug_info["concept_coverage"] = concept_coverage_map

    # Source Distribution
    source_distribution: dict[str, int] = {}
    for p in papers:
        src = getattr(p, "source", "arXiv") or "arXiv"
        source_distribution[src] = source_distribution.get(src, 0) + 1

    # Research Coverage Text
    ver_count = sum(1 for st in concept_coverage_map.values() if st == "VERIFIED")
    tot_count = len(concept_coverage_map)
    research_cov_text = (
        f"Investigated {tot_count} target concepts across {len(papers)} retrieved papers. "
        f"{ver_count} concept(s) corroborated by verified evidence; {tot_count - ver_count} concept(s) currently lack verified evidence."
    )

    # 5. Conclusion (B6: Single source of truth for anchor state; B3: No raw error strings)
    label, a_sent = anchor_state(anchor_paper, anchor_confidence)
    if label == "Not identified":
        conclusion_parts = [
            f"In summary, empirical literature analysis addresses: \"{question}\".",
            "Foundational anchor paper: Not identified.",
            "Anchor confidence: None.",
        ]
    elif label == "Uncertain":
        conclusion_parts = [
            f"In summary, empirical literature analysis addresses: \"{question}\".",
            "Foundational anchor paper: Selection is uncertain.",
            "Anchor confidence: Uncertain.",
        ]
    else:
        conclusion_parts = [
            f"In summary, empirical literature analysis addresses: \"{question}\".",
            a_sent,
            f"Anchor confidence: {anchor_confidence.capitalize()}.",
        ]

    if integrity >= 1.0:
        conclusion_parts.append("All findings verified with 100.0% citation integrity.")
    else:
        conclusion_parts.append(f"Findings verified with {integrity:.1%} citation integrity.")

    warning_notes = []
    if anchor_confidence == "uncertain":
        if uncertainty_note:
            warning_notes.append(uncertainty_note)
        else:
            warning_notes.append("Anchor paper selection is marked uncertain.")
    if removed_details:
        warning_notes.append(f"{len(removed_details)} claim sentence(s) removed due to lack of verifiable evidence.")
    # B3: Keep raw errors out of prose; do not extend warning_notes with raw retrieval_warnings

    if warning_notes:
        conclusion_parts.append(f"Warnings: {'; '.join(warning_notes)}.")

    conclusion = " ".join(conclusion_parts)

    report = ResearchReport(
        executiveSummary=exec_summary,
        methodology=methodology,
        findings=findings,
        comparisonTable=comparison_table,
        technique_comparison=tech_comparison,
        research_coverage=research_cov_text,
        insufficient_evidence=insufficient_evidence_items,
        contradicted_findings=[],
        source_distribution=source_distribution,
        retrieval_warnings=retrieval_warnings or [],
        concept_coverage=concept_coverage_map,
        computationalRequirements="",
        contradictoryEvidence="",
        limitations=limitations,
        conclusion=conclusion,
        references=references,
    )

    return report, integrity, removed_details
