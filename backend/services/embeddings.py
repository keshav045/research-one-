"""
Embeddings Service — Transformers Dense Embeddings
===================================================
Generates dense vector embeddings for queries and passages.
Supports BAAI/bge-small-en-v1.5 and sentence-transformers/all-MiniLM-L6-v2.
Ensures vectors are L2-normalized so that IndexFlatIP computes exact cosine similarity.

Optimizations:
1. Configurable PyTorch thread count via settings.TORCH_NUM_THREADS.
2. Length-sorted batching with padding waste elimination and exact order restoration.
3. Configurable max sequence length (default 128 for passages, 512 optional).
4. In-memory LRU cache for query embeddings (maxsize=256).
5. On-disk SQLite embedding cache keyed by sha256(model + max_len + text).
6. Optional dynamic int8 quantization (EMBED_QUANTIZE).
7. Optional OpenAI text-embedding-3-small backend with automatic local fallback.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..config import settings

logger = logging.getLogger(__name__)

_tokenizer = None
_model: Optional[object] = None
_active_model_name: Optional[str] = None

# ── Query LRU Cache ──────────────────────────────────────────────────────────
_query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
_QUERY_CACHE_MAXSIZE: int = 256


def clear_query_cache() -> None:
    """Clear in-memory LRU query cache."""
    _query_cache.clear()


# ── SQLite Disk Cache ────────────────────────────────────────────────────────
_cache_db_path: Optional[Path] = None


def _get_cache_db_conn() -> sqlite3.Connection:
    global _cache_db_path
    if _cache_db_path is None:
        cache_dir = Path(settings.PDF_CACHE_DIR)
        cache_dir.mkdir(parents=True, exist_ok=True)
        _cache_db_path = cache_dir / "embedding_cache.sqlite"
    conn = sqlite3.connect(str(_cache_db_path), timeout=15.0)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS embedding_cache (
            cache_key TEXT PRIMARY KEY,
            embedding_blob BLOB,
            dim INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    return conn


def clear_disk_cache() -> None:
    """Clear persistent SQLite embedding cache."""
    try:
        conn = _get_cache_db_conn()
        conn.execute("DELETE FROM embedding_cache")
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.warning("[EmbeddingsCache] Clear failed: %s", exc)


def _compute_cache_key(model_name: str, max_length: int, text: str) -> str:
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8"))
    h.update(str(max_length).encode("utf-8"))
    h.update(text[:1000].encode("utf-8"))
    return h.hexdigest()


def _lookup_disk_cache(keys: list[str]) -> dict[str, np.ndarray]:
    if not keys or not getattr(settings, "EMBED_CACHE_ENABLED", True):
        return {}
    try:
        conn = _get_cache_db_conn()
        cur = conn.cursor()
        found: dict[str, np.ndarray] = {}
        chunk_size = 500
        for i in range(0, len(keys), chunk_size):
            chunk = keys[i : i + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            cur.execute(
                f"SELECT cache_key, embedding_blob, dim FROM embedding_cache WHERE cache_key IN ({placeholders})",
                chunk,
            )
            for k, blob, dim in cur.fetchall():
                arr = np.frombuffer(blob, dtype=np.float32)
                if arr.size == dim:
                    found[k] = arr.copy()
        conn.close()
        return found
    except Exception as exc:
        logger.warning("[EmbeddingsCache] Read failed: %s", exc)
        return {}


def _save_disk_cache(entries: list[tuple[str, np.ndarray]]) -> None:
    if not entries or not getattr(settings, "EMBED_CACHE_ENABLED", True):
        return
    try:
        conn = _get_cache_db_conn()
        rows = [(k, arr.tobytes(), int(arr.size)) for k, arr in entries]
        conn.executemany(
            "INSERT OR REPLACE INTO embedding_cache (cache_key, embedding_blob, dim) VALUES (?, ?, ?)",
            rows,
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.warning("[EmbeddingsCache] Write failed: %s", exc)


# ── Model Initialization & Quantization ───────────────────────────────────────

def reset_model() -> None:
    """Reset loaded model singleton (used for tests or reconfiguration)."""
    global _tokenizer, _model, _active_model_name
    _tokenizer = None
    _model = None
    _active_model_name = None


def get_model():
    """Lazy-load the AutoModel tokenizer + model (first call only)."""
    global _tokenizer, _model, _active_model_name
    if _model is not None:
        return _tokenizer, _model

    from transformers import AutoTokenizer, AutoModel

    if hasattr(settings, "TORCH_NUM_THREADS") and settings.TORCH_NUM_THREADS > 0:
        try:
            torch.set_num_threads(settings.TORCH_NUM_THREADS)
        except Exception:
            pass

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

            if getattr(settings, "EMBED_QUANTIZE", False):
                try:
                    _model = torch.quantization.quantize_dynamic(
                        _model, {torch.nn.Linear}, dtype=torch.qint8
                    )
                    logger.info("[Embeddings] Model dynamically quantized to int8 (EMBED_QUANTIZE=True)")
                except Exception as q_exc:
                    logger.warning("[Embeddings] Dynamic quantization failed: %s", q_exc)

            logger.info("[Embeddings] Model ready: %s", model_name)
            return _tokenizer, _model
        except Exception as exc:
            logger.warning("[Embeddings] Failed loading %s: %s", model_name, exc)

    raise RuntimeError("Failed to load any embedding model from the candidate list")


def _pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean or CLS pooling depending on model family."""
    if _active_model_name and "bge" in _active_model_name.lower():
        # BGE models use first token (CLS)
        return token_embeddings[:, 0]
    # Sentence-transformers use mean pooling
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask_expanded, dim=1) / torch.clamp(mask_expanded.sum(dim=1), min=1e-9)


# ── OpenAI Embeddings Backend ─────────────────────────────────────────────────

def _embed_openai(texts: list[str]) -> Optional[np.ndarray]:
    """Call OpenAI text-embedding-3-small with dimensions=384 in batches of up to 100."""
    if not settings.is_openai_configured:
        return None
    try:
        import httpx
        url = "https://api.openai.com/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        all_vecs: list[np.ndarray] = []
        batch_size = 100
        with httpx.Client(timeout=30.0) as client:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                payload = {
                    "input": batch,
                    "model": "text-embedding-3-small",
                    "dimensions": 384,
                }
                resp = None
                for attempt in range(3):
                    try:
                        resp = client.post(url, json=payload, headers=headers)
                        resp.raise_for_status()
                        break
                    except Exception as err:
                        if attempt == 2:
                            raise err
                        time.sleep(0.5 * (2 ** attempt))

                if resp is None:
                    return None

                data = resp.json()["data"]
                data = sorted(data, key=lambda x: x["index"])
                vecs = np.array([x["embedding"] for x in data], dtype=np.float32)
                # L2 normalize
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1e-9, norms)
                vecs = vecs / norms
                all_vecs.append(vecs)
        return np.vstack(all_vecs)
    except Exception as exc:
        logger.warning("[Embeddings] OpenAI embeddings failed: %s; falling back to local model", exc)
        return None


# ── Core Embedding Functions ──────────────────────────────────────────────────

def embed_texts(
    texts: list[str],
    batch_size: int = 32,
    max_length: Optional[int] = None,
) -> np.ndarray:
    """
    Encode a list of strings into dense float32 L2-normalized embeddings.

    Features:
    - Optional OpenAI backend with graceful local fallback
    - Disk cache lookup by sha256(model_name + max_length + text)
    - Length sorting of uncached texts to eliminate padding waste
    - Exact order restoration to match input texts order
    """
    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    # 1. Check OpenAI backend if configured
    if getattr(settings, "EMBEDDING_BACKEND", "local").lower() == "openai":
        openai_embs = _embed_openai(texts)
        if openai_embs is not None:
            return openai_embs

    effective_max_length = (
        max_length if max_length is not None else getattr(settings, "EMBED_MAX_LENGTH", 128)
    )
    tokenizer, model = get_model()
    model_name = _active_model_name or settings.EMBEDDING_MODEL

    # 2. Check disk cache
    cache_keys = [_compute_cache_key(model_name, effective_max_length, t) for t in texts]
    cached_map = _lookup_disk_cache(cache_keys)

    miss_indices = [idx for idx, k in enumerate(cache_keys) if k not in cached_map]

    if not miss_indices:
        # All embeddings found in disk cache
        return np.vstack([cached_map[k] for k in cache_keys])

    uncached_texts = [texts[idx] for idx in miss_indices]

    # 3. Sort uncached texts by length to minimize padding waste
    sort_order = sorted(range(len(uncached_texts)), key=lambda i: len(uncached_texts[i]))
    sorted_texts = [uncached_texts[i] for i in sort_order]

    sorted_batch_embeddings: list[np.ndarray] = []
    for i in range(0, len(sorted_texts), batch_size):
        batch = [t[:1000] for t in sorted_texts[i : i + batch_size]]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=effective_max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(**inputs)
            pooled = _pool(outputs.last_hidden_state, inputs["attention_mask"])
            normalized = F.normalize(pooled, p=2, dim=1)
            sorted_batch_embeddings.append(normalized.cpu().numpy().astype(np.float32))

    sorted_embs = np.vstack(sorted_batch_embeddings)
    dim = sorted_embs.shape[1]

    # Restore uncached texts order
    uncached_embs = np.empty((len(uncached_texts), dim), dtype=np.float32)
    uncached_embs[sort_order] = sorted_embs

    # 4. Save newly computed embeddings to disk cache
    new_cache_entries = [
        (cache_keys[orig_idx], uncached_embs[i])
        for i, orig_idx in enumerate(miss_indices)
    ]
    _save_disk_cache(new_cache_entries)

    # 5. Assemble final array in exact original order
    final_embeddings = np.empty((len(texts), dim), dtype=np.float32)
    for i, k in enumerate(cache_keys):
        if k in cached_map:
            final_embeddings[i] = cached_map[k]
    for i, orig_idx in enumerate(miss_indices):
        final_embeddings[orig_idx] = uncached_embs[i]

    return final_embeddings


def embed_query(query: str, max_length: Optional[int] = None) -> np.ndarray:
    """
    Encode a single query string into a normalized 2-D array of shape (1, D).
    Uses an in-memory LRU cache (maxsize=256) keyed by query string.
    """
    q_key = query.strip()
    if q_key in _query_cache:
        _query_cache.move_to_end(q_key)
        return _query_cache[q_key].copy()

    # BGE models benefit from instruction prefix on queries
    q_text = query
    if _active_model_name and "bge" in _active_model_name.lower() and not query.startswith("Represent this"):
        q_text = f"Represent this sentence for searching relevant passages: {query}"

    emb = embed_texts([q_text], max_length=max_length)

    _query_cache[q_key] = emb.copy()
    if len(_query_cache) > _QUERY_CACHE_MAXSIZE:
        _query_cache.popitem(last=False)

    return emb
