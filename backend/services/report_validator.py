"""Report Validation Gate
======================
Validates synthesized research reports before saving or PDF export.
Catches content regressions, hallucinations, repeated text, out-of-range
citations, raw error strings, and anchor contradictions.
"""

from __future__ import annotations

import re
from ..models.schemas import ResearchReport
from .text_utils import split_into_sentences

RAW_ERROR_PATTERNS = [
    re.compile(r"\bHTTP\b", re.IGNORECASE),
    re.compile(r"\b429\b"),
    re.compile(r'all:"'),
    re.compile(r"\bTraceback\b", re.IGNORECASE),
]


def validate_report(report: ResearchReport) -> list[str]:
    """
    Validates a ResearchReport against quality and integrity invariants.
    Returns a list of validation failure descriptions (empty if valid).
    """
    problems: list[str] = []
    if not report:
        return ["Report object is None or empty."]

    # 1. Section text extraction
    section_texts: dict[str, list[str]] = {}
    if report.executiveSummary:
        section_texts["Executive Summary"] = [report.executiveSummary]
    
    findings_texts = []
    if report.findings:
        for idx, sec in enumerate(report.findings):
            sec_title = sec.sectionTitle.strip() if sec.sectionTitle else f"Section {idx+1}"
            if not sec.paragraphs:
                problems.append(f"Empty section heading with no content: '{sec_title}'")
            for p in sec.paragraphs:
                p_text = (p.text or "").strip()
                if not p_text:
                    problems.append(f"Empty paragraph in section '{sec_title}'")
                else:
                    findings_texts.append(p_text)
                    # Check claims without a citation in findings
                    if not p.citations and not re.search(r"\[\d+\]", p_text):
                        problems.append(f"Claim paragraph in '{sec_title}' has no citations: '{p_text[:60]}...'")
    section_texts["Findings"] = findings_texts

    if report.limitations:
        section_texts["Limitations"] = [str(l) for l in report.limitations]
    if report.conclusion:
        section_texts["Conclusion"] = [report.conclusion]

    # 2. Check for repeated paragraphs across sections
    para_seen = set()
    for sec_name, paras in section_texts.items():
        for p in paras:
            p_clean = " ".join(p.split()).lower()
            if len(p_clean) > 30:
                if p_clean in para_seen:
                    problems.append(f"Repeated paragraph detected: '{p[:70]}...'")
                para_seen.add(p_clean)

    # 3. Check for the same sentence longer than 40 chars in more than one section
    sentence_to_sections: dict[str, list[str]] = {}
    for sec_name, paras in section_texts.items():
        for p in paras:
            sentences = split_into_sentences(p)
            for s in sentences:
                s_norm = " ".join(re.sub(r"\[\d+\]", "", s).split()).strip().lower()
                if len(s_norm) > 40:
                    if s_norm not in sentence_to_sections:
                        sentence_to_sections[s_norm] = []
                    if sec_name not in sentence_to_sections[s_norm]:
                        sentence_to_sections[s_norm].append(sec_name)

    for s_norm, sections in sentence_to_sections.items():
        if len(sections) > 1:
            problems.append(
                f"Sentence longer than 40 characters appears in multiple sections ({', '.join(sections)}): '{s_norm[:70]}...'"
            )

    # 4. Check for citation markers outside 1..len(references)
    ref_count = len(report.references) if report.references else 0
    all_text = " ".join(
        [report.executiveSummary or ""]
        + findings_texts
        + [str(l) for l in report.limitations or []]
        + [report.conclusion or ""]
    )
    citation_markers = [int(m) for m in re.findall(r"\[(\d+)\]", all_text)]
    for n in citation_markers:
        if n < 1 or n > ref_count:
            problems.append(f"Citation marker [{n}] is outside valid reference range 1..{ref_count}")

    # 5. Check for raw error strings
    for pattern in RAW_ERROR_PATTERNS:
        matches = pattern.findall(all_text)
        if matches:
            problems.append(f"Raw error string detected in report text matching pattern '{pattern.pattern}'")

    # 6. Check for anchor contradiction
    conc_lower = (report.conclusion or "").lower()
    method_lower = (report.methodology or "").lower()
    exec_lower = (report.executiveSummary or "").lower()
    has_anchor_missing = any("not identified" in t for t in [conc_lower, method_lower, exec_lower])
    has_high_confidence = any("confidence: high" in t or "confidence is high" in t for t in [conc_lower, method_lower])
    if has_anchor_missing and has_high_confidence:
        problems.append("Anchor contradiction: anchor is not identified but confidence is stated as high.")

    return problems
