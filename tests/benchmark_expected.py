"""
Ground-truth benchmark expectations and strict programmatic scoring.
=====================================================================
Rule: "Found" means arXiv ID, DOI, or exact normalized title equality.
Never judge by similar titles or substring matches.
"""

from __future__ import annotations

import re
from typing import Any, Optional
from backend.services.ranker import normalize_paper_title


BENCHMARK_EXPECTED: dict[str, dict[str, Any]] = {
    "Transformer": {
        "question": "Which paper introduced the Transformer architecture, and what was its key idea?",
        "expected_arxiv": ["1706.03762"],
        "expected_titles": ["attention is all you need"],
        "expected_dois": ["10.48550/arxiv.1706.03762"],
    },
    "GAN": {
        "question": "Who introduced generative adversarial networks?",
        "expected_arxiv": ["1406.2661"],
        "expected_titles": [
            "generative adversarial networks",
            "generative adversarial nets",
        ],
        "expected_dois": ["10.48550/arxiv.1406.2661"],
    },
    "Adam": {
        "question": "Which paper proposed the Adam optimizer?",
        "expected_arxiv": ["1412.6980"],
        "expected_titles": ["adam a method for stochastic optimization"],
        "expected_dois": ["10.48550/arxiv.1412.6980"],
    },
    "RAG": {
        "question": "What is retrieval-augmented generation and who proposed it?",
        "expected_arxiv": ["2005.11401"],
        "expected_titles": [
            "retrieval augmented generation for knowledge intensive nlp tasks",
        ],
        "expected_dois": ["10.48550/arxiv.2005.11401"],
    },
    "Dropout": {
        "question": "Which paper introduced dropout?",
        # JMLR 2014; arXiv 1207.0580 is the earlier preprint
        "expected_arxiv": ["1207.0580"],
        "expected_titles": [
            "dropout a simple way to prevent neural networks from overfitting",
            "improving neural networks by preventing co adaptation of feature detectors",
        ],
        "expected_dois": ["10.48550/arxiv.1207.0580"],
    },
    "BatchNorm": {
        "question": "Who introduced batch normalization?",
        "expected_arxiv": ["1502.03167"],
        "expected_titles": [
            "batch normalization accelerating deep network training by reducing internal covariate shift",
        ],
        "expected_dois": ["10.48550/arxiv.1502.03167"],
    },
    "ViT": {
        "question": "Which paper introduced the Vision Transformer?",
        "expected_arxiv": ["2010.11929"],
        "expected_titles": [
            "an image is worth 16x16 words transformers for image recognition at scale",
        ],
        "expected_dois": ["10.48550/arxiv.2010.11929"],
    },
    "word2vec": {
        "question": "Which paper introduced word2vec?",
        "expected_arxiv": ["1301.3781"],
        "expected_titles": [
            "efficient estimation of word representations in vector space",
        ],
        "expected_dois": ["10.48550/arxiv.1301.3781"],
    },
}


def _extract_paper_arxiv(paper: Any) -> Optional[str]:
    """Extract clean arXiv ID from paper attributes."""
    # Check paper.doi
    doi = getattr(paper, "doi", None) or ""
    if "10.48550/arxiv." in doi.lower():
        raw = doi.lower().split("10.48550/arxiv.")[-1].strip()
        return re.sub(r"v\d+$", "", raw)

    # Check paper.id
    pid = getattr(paper, "id", "") or ""
    if pid.startswith("arxiv-"):
        raw = pid[len("arxiv-"):].strip()
        return re.sub(r"v\d+$", "", raw)

    return None


def is_expected_paper(paper: Any, topic: str) -> bool:
    """
    Strictly checks whether a Paper matches the expected paper for a topic.
    Returns True ONLY if:
      1. Clean arXiv ID strictly matches expected arXiv ID(s).
      2. Clean DOI matches expected DOI(s).
      3. Exact normalized title equals an expected normalized title.
    NO substring matching. NO partial matching.
    """
    spec = BENCHMARK_EXPECTED.get(topic)
    if not spec:
        return False

    # 1. Check arXiv ID equality
    p_arxiv = _extract_paper_arxiv(paper)
    if p_arxiv:
        for exp_a in spec.get("expected_arxiv", []):
            if p_arxiv.lower() == exp_a.lower():
                return True

    # 2. Check DOI equality
    p_doi = (getattr(paper, "doi", None) or "").strip().lower()
    if p_doi:
        if p_doi.startswith("https://doi.org/"):
            p_doi = p_doi[len("https://doi.org/"):].strip()
        elif p_doi.startswith("http://doi.org/"):
            p_doi = p_doi[len("http://doi.org/"):].strip()
        for exp_doi in spec.get("expected_dois", []):
            clean_exp_doi = exp_doi.lower().replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
            if p_doi == clean_exp_doi:
                return True

    # 3. Check EXACT normalized title equality
    p_title = getattr(paper, "title", "") or ""
    norm_p = normalize_paper_title(p_title)
    if norm_p:
        for exp_title in spec.get("expected_titles", []):
            norm_exp = normalize_paper_title(exp_title)
            if norm_p == norm_exp:
                return True

    return False
