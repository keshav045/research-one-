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
        "acceptable_alternates": ["1310.4546"],
        "expected_titles": [
            "efficient estimation of word representations in vector space",
        ],
        "expected_dois": ["10.48550/arxiv.1301.3781"],
    },
    "BERT": {
        "question": "Which paper introduced BERT?",
        "expected_arxiv": ["1810.04805"],
        "expected_titles": [
            "bert pre training of deep bidirectional transformers for language understanding",
            "bert pretraining of deep bidirectional transformers for language understanding",
        ],
        "expected_dois": ["10.48550/arxiv.1810.04805"],
    },
    "U-Net": {
        "question": "Which paper introduced U-Net?",
        "expected_arxiv": ["1505.04597"],
        "expected_titles": [
            "u net convolutional networks for biomedical image segmentation",
            "unet convolutional networks for biomedical image segmentation",
        ],
        "expected_dois": ["10.48550/arxiv.1505.04597"],
    },
    "Faster R-CNN": {
        "question": "Which paper introduced Faster R-CNN?",
        "expected_arxiv": ["1506.01497"],
        "expected_titles": [
            "faster r cnn towards real time object detection with region proposal networks",
        ],
        "expected_dois": ["10.48550/arxiv.1506.01497"],
    },
    "Latent Diffusion": {
        "question": "Which paper introduced High-Resolution Image Synthesis with Latent Diffusion Models?",
        "expected_arxiv": ["2112.10752"],
        "expected_titles": [
            "high resolution image synthesis with latent diffusion models",
        ],
        "expected_dois": ["10.48550/arxiv.2112.10752"],
    },
    "DenseNet": {
        "question": "Which paper introduced DenseNet?",
        "expected_arxiv": ["1608.06993"],
        "expected_titles": [
            "densely connected convolutional networks",
        ],
        "expected_dois": ["10.48550/arxiv.1608.06993"],
    },
    "Mask R-CNN": {
        "question": "Which paper introduced Mask R-CNN?",
        "expected_arxiv": ["1703.06870"],
        "expected_titles": [
            "mask r cnn",
            "mask rcnn",
        ],
        "expected_dois": ["10.48550/arxiv.1703.06870"],
    },
    "Graph Convolutional Networks": {
        "question": "Which paper introduced Graph Convolutional Networks?",
        "expected_arxiv": ["1609.02907"],
        "expected_titles": [
            "semi supervised classification with graph convolutional networks",
        ],
        "expected_dois": ["10.48550/arxiv.1609.02907"],
    },
    "T5": {
        "question": "Which paper introduced T5?",
        "expected_arxiv": ["1910.10683"],
        "expected_titles": [
            "exploring the limits of transfer learning with a unified text to text transformer",
        ],
        "expected_dois": ["10.48550/arxiv.1910.10683"],
    },
    "YOLO": {
        "question": "Which paper introduced YOLO?",
        "expected_arxiv": ["1506.02640"],
        "expected_titles": [
            "you only look once unified real time object detection",
            "you only look once unified realtime object detection",
        ],
        "expected_dois": ["10.48550/arxiv.1506.02640"],
    },
    "Layer Normalization": {
        "question": "Which paper introduced Layer Normalization?",
        "expected_arxiv": ["1607.06450"],
        "expected_titles": [
            "layer normalization",
        ],
        "expected_dois": ["10.48550/arxiv.1607.06450"],
    },
    "GPT-3": {
        "question": "Which paper introduced GPT-3?",
        "expected_arxiv": ["2005.14165"],
        "expected_titles": [
            "language models are few-shot learners",
            "language models are few shot learners",
        ],
        "expected_dois": ["10.48550/arxiv.2005.14165"],
    },
    "SimCLR": {
        "question": "Which paper introduced SimCLR?",
        "expected_arxiv": ["2002.05709"],
        "expected_titles": [
            "a simple framework for contrastive learning of visual representations",
        ],
        "expected_dois": ["10.48550/arxiv.2002.05709"],
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


def check_paper_match(paper: Any, topic: str) -> str:
    """
    Checks paper match against ground truth expectations for topic.
    Returns:
      - 'exact': Matches expected_arxiv, expected_dois, or expected_titles.
      - 'acceptable': Matches acceptable_alternates (e.g. 1310.4546 for word2vec).
      - 'no': Miss / no match.
    """
    if is_expected_paper(paper, topic):
        return "exact"

    spec = BENCHMARK_EXPECTED.get(topic)
    if not spec:
        return "no"

    # Check acceptable alternates
    p_arxiv = _extract_paper_arxiv(paper)
    if p_arxiv:
        for alt_a in spec.get("acceptable_alternates", []):
            if p_arxiv.lower() == alt_a.lower():
                return "acceptable"

    # Also check DOI / ID for alternate arXiv
    doi_val = (getattr(paper, "doi", "") or "").lower()
    for alt_a in spec.get("acceptable_alternates", []):
        if alt_a.lower() in doi_val:
            return "acceptable"

    return "no"
