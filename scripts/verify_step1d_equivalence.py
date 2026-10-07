"""
Verify Step 1d: Embeddings Equivalence & Cosine Similarity Benchmark
Tests:
1. 200 sample passages from a real paper.
2. Old implementation (no sorting, max_length=512) vs New implementation (length sorted, max_length=512):
   Verifies cosine similarity >= 0.999 (within 1e-3) and exact order match.
3. Cosine similarity between max_length=512 and max_length=128.
"""

import os
import sys
from pathlib import Path

# Ensure project root in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import json
import numpy as np
import torch
import torch.nn.functional as F

from backend.models.database import SessionLocal, PaperRecord
from backend.services.embeddings import get_model, embed_texts, _pool
from backend.config import settings


def old_embed_texts(texts: list[str], batch_size: int = 32, max_length: int = 512) -> np.ndarray:
    """Original un-optimized embedding function: original order, max_length=512."""
    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    tokenizer, model = get_model()
    all_embeddings: list[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch = [t[:1000] for t in texts[i : i + batch_size]]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(**inputs)
            pooled = _pool(outputs.last_hidden_state, inputs["attention_mask"])
            normalized = F.normalize(pooled, p=2, dim=1)
            all_embeddings.append(normalized.cpu().numpy().astype(np.float32))

    return np.vstack(all_embeddings)


def main():
    db = SessionLocal()
    recs = db.query(PaperRecord).all()
    sample_passages = []
    for r in recs:
        try:
            p_list = json.loads(r.passages_json or "[]")
            for p in p_list:
                if isinstance(p, dict) and "text" in p and len(p["text"]) > 60:
                    sample_passages.append(p["text"])
                if len(sample_passages) >= 200:
                    break
        except Exception:
            continue
        if len(sample_passages) >= 200:
            break
    db.close()

    assert len(sample_passages) >= 200, f"Only found {len(sample_passages)} passages"
    passages = sample_passages[:200]
    print(f"Loaded {len(passages)} real sample passages.")

    # Disable disk cache during this test to measure raw computation
    orig_cache = settings.EMBED_CACHE_ENABLED
    settings.EMBED_CACHE_ENABLED = False

    try:
        # 1. Compute old embeddings (max_length=512, original order)
        old_vecs_512 = old_embed_texts(passages, max_length=512)

        # 2. Compute new embeddings with max_length=512 (length sorted, order restored)
        new_vecs_512 = embed_texts(passages, max_length=512)

        # Cosine similarity between old_512 and new_512
        sims_512 = np.sum(old_vecs_512 * new_vecs_512, axis=1)
        mean_sim_512 = float(np.mean(sims_512))
        min_sim_512 = float(np.min(sims_512))
        max_diff_512 = float(np.max(np.abs(1.0 - sims_512)))

        print("\n--- TEST 1: Old vs New (max_length=512) ---")
        print(f"Mean cosine similarity: {mean_sim_512:.6f}")
        print(f"Min cosine similarity:  {min_sim_512:.6f}")
        print(f"Max cosine difference:  {max_diff_512:.6e}")
        assert max_diff_512 < 1e-3, f"Cosine difference too large: {max_diff_512}"
        print("=> PASS: Order restoration and length-sorted batching identical within 1e-3!")

        # 3. Compute new embeddings with max_length=128
        new_vecs_128 = embed_texts(passages, max_length=128)
        sims_128_vs_512 = np.sum(old_vecs_512 * new_vecs_128, axis=1)
        mean_sim_128 = float(np.mean(sims_128_vs_512))
        min_sim_128 = float(np.min(sims_128_vs_512))
        p95_sim_128 = float(np.percentile(sims_128_vs_512, 5))  # 5th percentile of similarity

        print("\n--- TEST 2: max_length=128 vs max_length=512 ---")
        print(f"Mean cosine similarity: {mean_sim_128:.6f}")
        print(f"5th percentile sim:     {p95_sim_128:.6f}")
        print(f"Min cosine similarity:  {min_sim_128:.6f}")
        print(f"Mean cosine difference: {1.0 - mean_sim_128:.6f}")
        print("=> Step 1d verification successfully completed!")

    finally:
        settings.EMBED_CACHE_ENABLED = orig_cache


if __name__ == "__main__":
    main()
