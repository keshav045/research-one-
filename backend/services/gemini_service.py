"""
LLM Report Generation Service
================================
Supports two providers — configured via LLM_PROVIDER in .env:

  • "qwen"   — Alibaba Qwen via DashScope (OpenAI-compatible API) [default]
  • "gemini" — Google Gemini API

Both providers produce identical structured JSON research reports.
Qwen uses the openai SDK pointed at DashScope's base URL.
Gemini uses the google-generativeai SDK.

Model chains:
  Qwen   : qwen-plus → qwen-turbo → qwen-long
  Gemini : gemini-2.5-flash → gemini-2.0-flash → gemini-1.5-flash-latest
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Optional

from ..config import settings
from ..models.schemas import Paper, ResearchDepth, ResearchReport, ResearchSource

logger = logging.getLogger(__name__)

# ── Model chains ──────────────────────────────────────────────────────────────

OPENAI_MODEL_CHAIN = [
    "gpt-4o-mini",
    "gpt-4o",
]

QWEN_MODEL_CHAIN = [
    "qwen-plus",
    "qwen-turbo",
    "qwen-long",
]

GEMINI_MODEL_CHAIN = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash-latest",
]

_RETRYABLE_CODES = {"[429]", "[503]", "429", "503"}

# ─── Question Type Classifier ─────────────────────────────────────────────────

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


def _classify_question_type(question: str) -> str:
    is_factual     = bool(_FACTUAL_PATTERNS.search(question))
    is_comparative = bool(_COMPARATIVE_PATTERNS.search(question))
    if is_factual and not is_comparative:
        return "factual_lookup"
    if is_comparative:
        return "comparative_review"
    return "comparative_review"


# ─── OpenAI API ───────────────────────────────────────────────────────────────

async def _call_openai(model_name: str, prompt: str, retries: int = 2) -> str:
    """Async wrapper for OpenAI API via the openai SDK."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    for attempt in range(retries + 1):
        try:
            result = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are ResearchLens, an academic AI. "
                            "Always respond with valid JSON and nothing else — no markdown fences."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            return result.choices[0].message.content.strip()
        except Exception as exc:
            msg = str(exc)
            is_retryable = any(code in msg for code in _RETRYABLE_CODES)
            if is_retryable and attempt < retries:
                wait = (attempt + 1) * 3
                logger.warning("[OpenAI] %s attempt %d failed (%s). Retrying in %ds…", model_name, attempt + 1, msg[:80], wait)
                await asyncio.sleep(wait)
            else:
                raise


# ─── Qwen API (OpenAI-compatible) ─────────────────────────────────────────────

async def _call_qwen(model_name: str, prompt: str, retries: int = 2) -> str:
    """Async wrapper for Qwen DashScope via the openai SDK."""
    from openai import OpenAI

    client = OpenAI(
        api_key=settings.QWEN_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    for attempt in range(retries + 1):
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are ResearchLens, an academic AI. "
                                "Always respond with valid JSON and nothing else — no markdown fences."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2,
                ),
            )
            return result.choices[0].message.content.strip()
        except Exception as exc:
            msg = str(exc)
            is_retryable = any(code in msg for code in _RETRYABLE_CODES)
            if is_retryable and attempt < retries:
                wait = (attempt + 1) * 3
                logger.warning("[Qwen] %s attempt %d failed (%s). Retrying in %ds…", model_name, attempt + 1, msg[:80], wait)
                await asyncio.sleep(wait)
            else:
                raise


# ─── Gemini API ───────────────────────────────────────────────────────────────

async def _call_gemini(model_name: str, prompt: str, retries: int = 2) -> str:
    """Async wrapper for the Google Gemini SDK."""
    import google.generativeai as genai

    genai.configure(api_key=settings.GEMINI_API_KEY)
    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config=genai.GenerationConfig(response_mime_type="application/json"),
    )

    for attempt in range(retries + 1):
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: model.generate_content(prompt)
            )
            return result.text.strip()
        except Exception as exc:
            msg = str(exc)
            is_retryable = any(code in msg for code in _RETRYABLE_CODES)
            if is_retryable and attempt < retries:
                wait = (attempt + 1) * 3
                logger.warning("[Gemini] %s attempt %d failed (%s). Retrying in %ds…", model_name, attempt + 1, msg[:80], wait)
                await asyncio.sleep(wait)
            else:
                raise


# ─── Prompt Building ──────────────────────────────────────────────────────────

def _build_paper_block(papers: list[Paper], target_count: int) -> str:
    if not papers:
        return ""
    lines = ["\n\nREAL PAPERS RETRIEVED FROM arXiv / Semantic Scholar (USE THESE AS PRIMARY REFERENCES):"]
    for i, p in enumerate(papers[:target_count], 1):
        authors = ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else "")
        abstract_snippet = p.abstract[:500].replace("\n", " ")
        lines.append(
            f"\n[paper-{i}] ID=paper-{i}\n"
            f"  Title:    {p.title}\n"
            f"  Authors:  {authors}\n"
            f"  Year:     {p.publicationYear}\n"
            f"  Journal:  {p.journalConference}\n"
            f"  DOI:      {p.doi}\n"
            f"  Abstract: {abstract_snippet}…"
        )
    return "\n".join(lines)


def _build_prompt(
    question: str,
    depth: ResearchDepth,
    sources: list[ResearchSource],
    papers: list[Paper],
) -> str:
    target = {"Quick": 6, "Standard": 8, "Deep": 12}.get(depth.value, 8)
    source_str = ", ".join(s.value for s in sources)
    paper_block = _build_paper_block(papers, target)
    question_type = _classify_question_type(question)

    grounding = (
        f"- GROUNDING: Your 'references' MUST mirror the real papers above. "
        f"Use the exact IDs (paper-1…paper-{min(len(papers), target)}), titles, authors, years, DOIs. "
        f"Synthesise findings that accurately reflect what those papers likely contain "
        f"based on their titles and abstracts."
        if papers
        else f"- Include at least {target} distinct real academic papers in 'references'."
    )

    if question_type == "factual_lookup":
        methodology_schema = (
            "### Database Search Queries\\n"
            "- arXiv: (specific title or author name used to retrieve the source paper)\\n\\n"
            "### Inclusion Criteria\\n"
            "1. The original paper that first introduced the concept or architecture.\\n"
            "2. Peer-reviewed venue or preprint with high citation count (>100).\\n"
            "3. Directly answers the specific factual question posed.\\n\\n"
            "### Exclusion Criteria\\n"
            "1. Secondary surveys or tutorials that only summarise the original work.\\n"
            "2. Papers that merely cite the target paper without adding primary content.\\n\\n"
            "### Evidence Verification\\n"
            "Claims verified by NLI entailment against source passages from the original paper."
        )
    else:
        methodology_schema = (
            "### Database Search Queries\\n"
            "- arXiv: (keyword queries used to retrieve the papers above)\\n\\n"
            "### Inclusion Criteria\\n"
            "1. Peer-reviewed papers or recognised preprints with reproducible code or checkpoints.\\n"
            "2. Empirical evaluations on standard benchmark datasets.\\n"
            "3. Direct ablation studies reporting memory, latency, and quality metrics.\\n\\n"
            "### Exclusion Criteria\\n"
            "1. Theoretical commentary without empirical verification.\\n"
            "2. Proprietary closed-source architectures without architectural disclosures.\\n"
            "3. Unverified claims from informal whitepapers.\\n\\n"
            "### Evidence Verification\\n"
            "NLI-based entailment checking against source passages."
        )

    n_papers_actual = len(papers)
    counts_block = (
        f"\n\nCOUNTS (authoritative — do NOT contradict these in any text field):\n"
        f"  papers_retrieved = {n_papers_actual}\n"
        f"  verified_claims  = computed by NLI after you respond; do NOT guess or invent a number\n"
        f"  citation_coverage = computed by NLI after you respond; do NOT guess or invent a number\n"
    ) if papers else ""

    return f"""You are ResearchLens, an academic AI synthesizing scholarly literature with rigorous citation integrity.

Research question: "{question}"
Parameters: {depth.value} depth (~{target} papers), Sources: {source_str}{paper_block}{counts_block}

Return ONLY a valid JSON object matching this schema:
{{
  "executiveSummary": "2-3 academic paragraphs. MUST state exactly {n_papers_actual if papers else target} papers were retrieved (match papers_retrieved above). Do NOT mention any count of verified claims or citation coverage — those are computed by NLI after you respond and will appear in the header badges.",
  "methodology": "{methodology_schema}",
  "findings": [
    {{
      "sectionTitle": "<Specific title>",
      "paragraphs": [
        {{
          "text": "<Paragraph with empirical claims>",
          "citations": [
            {{
              "id": "cite-1",
              "badgeNumber": 1,
              "claim": "<Specific factual claim>",
              "status": "unverified",
              "paperId": "paper-1",
              "paperTitle": "<Title from references>",
              "authors": "<FirstAuthor et al.>",
              "year": 2024,
              "page": 4,
              "passage": "<Quoted sentence from the paper>",
              "highlightSentence": "<Key snippet>",
              "reason": "Confirmed in experimental table."
            }}
          ]
        }}
      ]
    }}
  ],
  "comparisonTable": [
    {{
      "model": "<Technique name>",
      "architectureType": "<Category>",
      "dataset": "<Benchmark>",
      "f1Score": "<Quality metric>",
      "mapScore": "<Compression ratio>",
      "fpsThroughput": "<Speedup>",
      "parametersM": "<Memory>",
      "gflops": "<Compute cost>",
      "citationId": "cite-1"
    }}
  ],
  "computationalRequirements": "<Hardware analysis>",
  "contradictoryEvidence": "<Conflicting findings>",
  "limitations": ["<Limitation 1>", "<Limitation 2>", "<Limitation 3>"],
  "conclusion": "DECISION FRAMEWORK: 1. Edge/memory-constrained: ... 2. Server batching: ... 3. Domain adaptation: ...",
  "references": [
    {{
      "id": "paper-1",
      "title": "<Full title>",
      "authors": ["<Author 1>", "<Author 2>"],
      "publicationYear": 2024,
      "journalConference": "<Venue>",
      "doi": "<DOI>",
      "source": "arXiv",
      "abstract": "<Abstract>",
      "claimsSupportedCount": 4,
      "evidenceCount": 3
    }}
  ]
}}

INSTRUCTIONS:
{grounding}
- At least 3 finding sections, each with 1-2 paragraphs and inline citations.
- At least 4 comparison table rows.
- Epistemic modesty: use "suggests", "indicates", "provides evidence for" instead of "proves", "demonstrates".
- Include explicit search queries, inclusion/exclusion criteria in methodology.
- Conclusion must have a clear per-use-case decision framework.
- FINDINGS PARAGRAPHS: every sentence must be directly supported by a citation in that paragraph.
- If a claim cannot be grounded in the retrieved papers, omit it entirely rather than writing a vague unsupported paraphrase.
- CRITICAL: The executiveSummary MUST reference exactly {n_papers_actual if papers else target} papers (the count in COUNTS above). Never write a different number."""


# ─── Unified Entry Point ──────────────────────────────────────────────────────

def _merge_real_papers(data: dict, papers: list[Paper]) -> dict:
    """Overlay real paper metadata (authors, year, DOI, pdfUrl) over LLM-generated references."""
    if not papers:
        return data
    paper_map = {p.id: p for p in papers}
    merged_refs = []
    for ref_data in data.get("references", []):
        real = paper_map.get(ref_data.get("id", ""))
        if real:
            ref_data["authors"]           = real.authors
            ref_data["publicationYear"]   = real.publicationYear
            ref_data["journalConference"] = real.journalConference
            ref_data["doi"]               = real.doi
            ref_data["source"]            = real.source
            if real.pdfUrl:
                ref_data["pdfUrl"] = real.pdfUrl
        merged_refs.append(ref_data)
    data["references"] = merged_refs
    return data


async def _try_models(
    model_chain: list[str],
    call_fn,
    prompt: str,
    papers: list[Paper],
    provider_name: str,
) -> Optional[ResearchReport]:
    """Try each model in the chain and return the first valid report."""
    for model_name in model_chain:
        try:
            logger.info("[%s] Trying model: %s (grounded=%s)", provider_name, model_name, bool(papers))
            raw = await call_fn(model_name, prompt)
            # Strip markdown fences if present
            raw = re.sub(r"^```(?:json)?\n?", "", raw).rstrip("`").strip()
            data = json.loads(raw)

            if not data.get("references") or not data.get("findings"):
                logger.warning("[%s] Incomplete response from %s", provider_name, model_name)
                continue

            data = _merge_real_papers(data, papers)
            report = ResearchReport(**data)
            logger.info("[%s] Success with %s (%d papers, %d table rows)",
                        provider_name, model_name, len(report.references), len(report.comparisonTable))
            return report

        except json.JSONDecodeError as exc:
            logger.warning("[%s] JSON decode error with %s: %s", provider_name, model_name, exc)
        except Exception as exc:
            logger.warning("[%s] %s failed: %s", provider_name, model_name, str(exc)[:120])

    logger.error("[%s] All models in fallback chain exhausted", provider_name)
    return None


async def generate_report_with_gemini(
    question: str,
    depth: ResearchDepth,
    sources: list[ResearchSource],
    papers: list[Paper],
) -> Optional[ResearchReport]:
    """
    Main entry point (name kept for backward compatibility with research_workflow.py).
    Routes to:
      1. Local HuggingFace LLM  (LLM_PROVIDER=local, no API key needed)
      2. Qwen DashScope API     (LLM_PROVIDER=qwen)
      3. Gemini API             (LLM_PROVIDER=gemini, or as last fallback)
    """
    prompt = _build_prompt(question, depth, sources, papers)
    provider = settings.LLM_PROVIDER.lower()

    # ── Option 1: OpenAI API ──────────────────────────────────────────────────
    if provider == "openai" or settings.is_openai_configured:
        if settings.is_openai_configured:
            report = await _try_models(OPENAI_MODEL_CHAIN, _call_openai, prompt, papers, "OpenAI")
            if report:
                return report
            logger.warning("[LLM] OpenAI chain exhausted, trying fallbacks…")

    # ── Option 2: Gemini API ──────────────────────────────────────────────────
    if settings.is_gemini_configured:
        report = await _try_models(GEMINI_MODEL_CHAIN, _call_gemini, prompt, papers, "Gemini")
        if report:
            return report
        logger.warning("[LLM] Gemini chain exhausted, trying Qwen…")

    # ── Option 3: Qwen API ────────────────────────────────────────────────────
    if settings.is_qwen_configured:
        report = await _try_models(QWEN_MODEL_CHAIN, _call_qwen, prompt, papers, "Qwen")
        if report:
            return report

    logger.error("[LLM] All providers failed to generate a report")
    return None


# ─── Simple Text Generation (used by local_llm_service._call_llm) ─────────────

async def generate_text_simple(prompt: str, max_tokens: int = 512) -> str:
    """
    Lightweight text generation that routes to Qwen or Gemini cloud API.
    Returns empty string on failure.
    """
    provider = settings.LLM_PROVIDER.lower()
    try:
        if (provider == "openai" or settings.is_openai_configured) and settings.is_openai_configured:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            resp = await client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.3,
            )
            return resp.choices[0].message.content or ""
        if provider == "qwen" and settings.is_qwen_configured:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(
                api_key=settings.QWEN_API_KEY,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            resp = await client.chat.completions.create(
                model="qwen-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.3,
            )
            return resp.choices[0].message.content or ""
        if settings.is_gemini_configured:
            import google.generativeai as genai
            genai.configure(api_key=settings.GEMINI_API_KEY)
            model = genai.GenerativeModel("gemini-2.0-flash")
            resp = await asyncio.to_thread(model.generate_content, prompt)
            return resp.text or ""
    except Exception as exc:
        logger.warning("[generate_text_simple] Failed: %s", exc)
    return ""
