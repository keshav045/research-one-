"""
Text Utilities for Sentence Splitting and Normalization
======================================================
Provides abbreviation-aware sentence splitting that prevents false splits on:
- Author initials ('Diederik P. Kingma', 'J. L. Ba')
- Academic abbreviations ('et al.', 'Fig.', 'Eq.', 'Sec.', 'No.', 'vs.', 'e.g.', 'i.e.', 'Ref.')
- Numeric decimals ('3.14', '0.80')
"""

import re

_SENTINEL = "\uE000"  # Private use unicode character

# Common academic & title abbreviations that end with a period
_ABBREVIATIONS = (
    r"\b(?:"
    r"et\s+al|"
    r"fig|figs|figure|figures|"
    r"eq|eqs|equation|equations|"
    r"sec|secs|section|sections|"
    r"no|nos|"
    r"vs|"
    r"ref|refs|"
    r"dr|prof|mr|mrs|ms|"
    r"approx|appr|ca|dept|"
    r"vol|vols|pp|p"
    r")\."
)


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into clean sentences, respecting abbreviations, initials, citations, and numbers.
    """
    if not text or not text.strip():
        return []

    clean = text.strip()

    # 1. Protect e.g. and i.e.
    clean = re.sub(r"\b([eE])\.([gG])\.", rf"\1{_SENTINEL}\2{_SENTINEL}", clean)
    clean = re.sub(r"\b([iI])\.([eE])\.", rf"\1{_SENTINEL}\2{_SENTINEL}", clean)

    # 2. Protect author initials like 'P.' in 'Diederik P. Kingma' or 'J. L. Ba'
    # Capital letter followed by period, followed by whitespace and another word
    clean = re.sub(r"(?<=\b[A-Za-z])\.(?=\s+[A-Za-z])", _SENTINEL, clean)

    # 3. Protect academic and general abbreviations
    clean = re.sub(_ABBREVIATIONS, lambda m: m.group(0)[:-1] + _SENTINEL, clean, flags=re.IGNORECASE)

    # 4. Protect decimals in numbers like 3.14 or 0.80
    clean = re.sub(r"(?<=\d)\.(?=\d)", _SENTINEL, clean)

    # 5. Split on sentence boundaries: period/question mark/exclamation followed by space and capital/number/quote/citation
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])", clean)

    # 6. Restore protected periods and clean
    results: list[str] = []
    for part in parts:
        restored = part.replace(_SENTINEL, ".").strip()
        if restored:
            results.append(restored)

    return results
