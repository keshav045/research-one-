"""
Embeddings Service — Transformers Dense Embeddings
===================================================
Generates dense vector embeddings for queries and passages.
Supports BAAI/bge-small-en-v1.5 and sentence-transformers/all-MiniLM-L6-v2.
Ensures vectors are L2-normalized so that IndexFlatIP computes exact cosine similarity.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..config import settings

logger = logging.getLogger(__name__)

_tokenizer = None
_model: Optional[object] = None
_active_model_name: Optional[str] = None


def get_model():
    """Lazy-load the AutoModel tokenizer + model (first call only)."""
    global _tokenizer, _model, _active_model_name
    if _model is not None:
        return _tokenizer, _model

    from transformers import AutoTokenizer, AutoModel

    models_to_try = [
        settings.EMBEDDING_MODEL,
        "BAAI/bge-small-en-v1.5",
        "sentence-transformers/all-MiniLM-L6-v2",
        "all-MiniLM-L6-v2",
    ]

    for model_name in dict.fromkeys(models_to_try):
        try:
            logger.info("[Embeddings] Trying embedding model: %s", model_name)
            _tokenizer = AutoTokenizer.from_pretrained(model_name)
            _model = AutoModel.from_pretrained(model_name)
            _model.eval()
            _active_model_name = model_name
            logger.info("[Embeddings] Model ready: %s", model_name)
            return _tokenizer, _model
        except Exception as exc:
            logger.warning("[Embeddings] Failed loading %s: %s", model_name, exc)

    raise RuntimeError("Failed to load any embedding model from the candidate list")


def _pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean or CLS pooling depending on model family."""
    global _active_model_name
    if _active_model_name and "bge" in _active_model_name.lower():
        # BGE models use first token (CLS)
        return token_embeddings[:, 0]
    # Sentence-transformers use mean pooling
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask_expanded, dim=1) / torch.clamp(mask_expanded.sum(dim=1), min=1e-9)


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """
    Encode a list of strings into dense float32 L2-normalized embeddings.
    """
    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    tokenizer, model = get_model()
    all_embeddings: list[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch = [t[:1000] for t in texts[i : i + batch_size]]

        # BGE query instruction prefix is only for queries, passages use raw text
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(**inputs)
            pooled = _pool(outputs.last_hidden_state, inputs["attention_mask"])
            normalized = F.normalize(pooled, p=2, dim=1)
            all_embeddings.append(normalized.cpu().numpy().astype(np.float32))

    return np.vstack(all_embeddings)


def embed_query(query: str) -> np.ndarray:
    """
    Encode a single query string into a normalized 2-D array of shape (1, D).
    """
    global _active_model_name
    # BGE models benefit from instruction prefix on queries
    q_text = query
    if _active_model_name and "bge" in _active_model_name.lower() and not query.startswith("Represent this"):
        q_text = f"Represent this sentence for searching relevant passages: {query}"

    emb = embed_texts([q_text])
    return emb
