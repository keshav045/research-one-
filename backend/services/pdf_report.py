"""PDF export for ResearchLens reports (PyMuPDF).

Kept separate from streamlit_app.py so it can be unit-tested without Streamlit/torch.
Rule: every wrapped line is drawn exactly ONCE.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

from ..models.schemas import ResearchInvestigation

logger = logging.getLogger(__name__)

# ── PDF text helpers (single-draw wrapping, WinAnsi-safe text) ────────────────
_PDF_CHAR_MAP = {
    "\u2022": "-", "\u00b7": "-", "\u2014": "-", "\u2013": "-", "\u2212": "-",
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2026": "...", "\u00a0": " ", "\u2192": "->", "\u2265": ">=", "\u2264": "<=",
}


def _pdf_safe(text: object) -> str:
    """Return text the built-in Helvetica font can draw (no '?' glyphs)."""
    import unicodedata

    out = str(text if text is not None else "")
    for src, dst in _PDF_CHAR_MAP.items():
        out = out.replace(src, dst)
    out = unicodedata.normalize("NFKD", out)
    return out.encode("latin-1", "ignore").decode("latin-1")


def _pdf_wrap(text: str, max_w: float, fontsize: float, fontname: str = "helv") -> list[str]:
    """Greedy word-wrap measured with real font metrics. Returns finished lines."""
    import fitz

    def width(s: str) -> float:
        return fitz.get_text_length(s, fontname=fontname, fontsize=fontsize)

    lines: list[str] = []
    for raw in str(text).split("\n"):
        current = ""
        for word in raw.split():
            while width(word) > max_w:  # very long token (URL/DOI): hard split
                cut = len(word)
                while cut > 1 and width(word[:cut]) > max_w:
                    cut -= 1
                if current:
                    lines.append(current)
                    current = ""
                lines.append(word[:cut])
                word = word[cut:]
            candidate = f"{current} {word}".strip()
            if current and width(candidate) > max_w:
                lines.append(current)  # flush ONCE per finished line
                current = word
            else:
                current = candidate
        lines.append(current)
    return lines


def _pdf_ellipsize(text: str, max_w: float, fontsize: float, fontname: str = "helv") -> str:
    import fitz

    if fitz.get_text_length(text, fontname=fontname, fontsize=fontsize) <= max_w:
        return text
    while text and fitz.get_text_length(text + "...", fontname=fontname, fontsize=fontsize) > max_w:
        text = text[:-1]
    return text.rstrip() + "..."


def build_pdf_report(inv: ResearchInvestigation, validate: bool = True) -> bytes:
    """Construct an academic-quality, publication-ready PDF document using PyMuPDF."""
    try:
        import fitz
    except ImportError:
        logger.error("PyMuPDF (fitz) is not installed; cannot generate PDF.")
        return b""

    if validate and inv.report:
        from .report_validator import validate_report
        val_errors = validate_report(inv.report)
        if val_errors:
            logger.error("[PDF] Report validation failed; refusing to generate PDF: %s", val_errors)
            return b""

    doc = fitz.open()
    page_w, page_h = 595.3, 841.9  # A4 size
    margin = 50.0
    content_w = page_w - 2 * margin
    content_bottom = page_h - margin

    def add_blank_page():
        return doc.new_page(width=page_w, height=page_h)

    current_page = add_blank_page()
    y_cursor = margin

    def check_space(needed_h: float):
        nonlocal current_page, y_cursor
        if y_cursor + needed_h > content_bottom - 24:
            current_page = add_blank_page()
            y_cursor = margin

    # Header Card on Page 1
    banner_h = 80.0
    current_page.draw_rect(
        fitz.Rect(margin, y_cursor, margin + content_w, y_cursor + banner_h),
        color=None,
        fill=(0.06, 0.10, 0.18),  # Deep navy / slate
    )
    current_page.insert_text(
        (margin + 16, y_cursor + 24),
        "ResearchLens - Autonomous Academic Synthesis",
        fontsize=13,
        fontname="helv",
        color=(1.0, 1.0, 1.0),
    )
    q_str = inv.question if len(inv.question) < 85 else inv.question[:82] + "..."
    current_page.insert_text(
        (margin + 16, y_cursor + 44),
        _pdf_safe(f"Research Question: {q_str}"),
        fontsize=9.0,
        fontname="helv",
        color=(0.85, 0.90, 0.98),
    )
    created_str = inv.createdAt[:10] if inv.createdAt else datetime.now(timezone.utc).strftime("%Y-%m-%d")
    meta_line = (
        f"Date: {created_str}   |   Confidence: {inv.research_confidence}   |   "
        f"Citation Integrity: {inv.citation_integrity}%   |   Papers Analyzed: {inv.papersAnalyzed}"
    )
    current_page.insert_text(
        (margin + 16, y_cursor + 63),
        meta_line,
        fontsize=8.0,
        fontname="helv",
        color=(0.60, 0.72, 0.88),
    )
    y_cursor += banner_h + 20.0

    if not inv.report:
        rect = fitz.Rect(margin, y_cursor, margin + content_w, content_bottom)
        current_page.insert_textbox(rect, f"Status: {inv.status}\n\nNo synthesized report generated.", fontsize=10)
        return doc.tobytes()

    rep = inv.report

    def render_heading(title: str, level: int = 1):
        nonlocal y_cursor
        check_space(34.0 if level == 1 else 26.0)
        font_size = 12.5 if level == 1 else 10.5
        current_page.insert_text(
            (margin, y_cursor + (13 if level == 1 else 10)),
            _pdf_safe(title),
            fontsize=font_size,
            fontname="helv",
            color=(0.06, 0.10, 0.18) if level == 1 else (0.18, 0.24, 0.36),
        )
        if level == 1:
            current_page.draw_line(
                fitz.Point(margin, y_cursor + 18),
                fitz.Point(margin + content_w, y_cursor + 18),
                color=(0.82, 0.86, 0.92),
                width=0.8,
            )
            y_cursor += 28.0
        else:
            y_cursor += 20.0

    def render_paragraph(text: str):
        """Draw each wrapped line exactly ONCE (never inside a per-word loop)."""
        nonlocal current_page, y_cursor
        text = _pdf_safe(text)
        # Never print a [N] that has no entry in the reference list.
        n_refs = len(rep.references or [])
        text = re.sub(r"\[(\d+)\]", lambda m: m.group(0) if 1 <= int(m.group(1)) <= n_refs else "", text)
        text = re.sub(r"[ \t]+([.,;:])", r"\1", text)
        if not text.strip():
            return
        size, leading = 9.0, 12.5
        for line in _pdf_wrap(text, content_w, size):
            if y_cursor + leading > content_bottom - 24:
                current_page = add_blank_page()
                y_cursor = margin
            current_page.insert_text(
                (margin, y_cursor + size), line,
                fontsize=size, fontname="helv", color=(0.12, 0.16, 0.24),
            )
            y_cursor += leading
        y_cursor += 6.0

    # 1. Executive Summary
    if rep.executiveSummary:
        render_heading("Executive Summary")
        render_paragraph(rep.executiveSummary)

    # 2. Methodological Comparison Table
    if rep.comparisonTable:
        render_heading("Methodological Comparison")
        col_widths = [215.0, 40.0, 165.0, 75.3]
        row_h = 18.0
        headers = ["Paper", "Year", "Venue", "Citations"]
        check_space(row_h * (len(rep.comparisonTable) + 2))
        x = margin
        for idx, h_text in enumerate(headers):
            cell_w = col_widths[idx]
            current_page.draw_rect(fitz.Rect(x, y_cursor, x + cell_w, y_cursor + row_h), color=(0.8, 0.85, 0.9), fill=(0.94, 0.96, 0.98))
            current_page.insert_text((x + 4, y_cursor + 12), h_text, fontsize=7.5, fontname="helv", color=(0.15, 0.2, 0.3))
            x += cell_w
        y_cursor += row_h

        for r_idx, row in enumerate(rep.comparisonTable):
            check_space(row_h + 4)
            x = margin
            # Use the real metadata fields; legacy dataset/f1Score/mapScore hold year/venue/citations.
            vals = [
                row.title or row.model,
                row.year or "Not reported",
                row.venue or "Not reported",
                row.citationCount or "Not reported",
            ]
            bg_col = (1.0, 1.0, 1.0) if r_idx % 2 == 0 else (0.98, 0.98, 0.99)
            for idx, val_text in enumerate(vals):
                cell_w = col_widths[idx]
                shown = _pdf_ellipsize(_pdf_safe(val_text), cell_w - 8, 7.5)
                current_page.draw_rect(fitz.Rect(x, y_cursor, x + cell_w, y_cursor + row_h), color=(0.85, 0.88, 0.92), fill=bg_col)
                current_page.insert_text((x + 4, y_cursor + 12), shown, fontsize=7.5, fontname="helv", color=(0.2, 0.25, 0.35))
                x += cell_w
            y_cursor += row_h
        y_cursor += 12.0

    # 3. Detailed Research Findings
    if rep.findings:
        render_heading("Detailed Research Findings")
        for sec in rep.findings:
            render_heading(sec.sectionTitle, level=2)
            for p in sec.paragraphs:
                render_paragraph(p.text)  # citation markers are already inside p.text

    # 4. Limitations
    if rep.limitations:
        render_heading("Limitations & Boundary Conditions")
        for lim in rep.limitations:
            render_paragraph(f"- {lim}")

    # 5. Conclusion
    if rep.conclusion:
        render_heading("Conclusion")
        render_paragraph(rep.conclusion)

    # 6. References
    if rep.references:
        render_heading("References")
        for idx, ref in enumerate(rep.references, 1):
            authors_fmt = ", ".join(ref.authors[:3]) + (" et al" if len(ref.authors) > 3 else "")
            ref_line = f"[{idx}] {ref.title} ({ref.publicationYear}). {authors_fmt}. {ref.journalConference or ref.source}."
            if ref.doi:
                ref_line += f" DOI: {ref.doi}"
            render_paragraph(ref_line)

    # Running Footer on all pages
    total_pages = len(doc)
    for p_no in range(total_pages):
        p = doc[p_no]
        footer_text = f"ResearchLens Autonomous Academic Assistant   |   Page {p_no + 1} of {total_pages}"
        p.draw_line(fitz.Point(margin, page_h - 32), fitz.Point(page_w - margin, page_h - 32), color=(0.88, 0.90, 0.94), width=0.5)
        p.insert_text((margin, page_h - 20), footer_text, fontsize=7.5, fontname="helv", color=(0.55, 0.60, 0.68))

    return doc.tobytes()
