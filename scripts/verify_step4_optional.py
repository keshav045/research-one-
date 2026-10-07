"""
Step 4 Verification:
4a. Dynamic int8 quantization: evaluate top-5 passage retrieval overlap, cosine similarity, speed.
4b. OpenAI embeddings backend test and cost/speed estimation.
"""

import os
import sys
import time
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import json
import numpy as np
import torch

from backend.config import settings
from backend.models.database import SessionLocal, PaperRecord
from backend.services.embeddings import (
    reset_model,
    get_model,
    embed_texts,
    embed_query,
    _embed_openai,
)


def run_quantization_evaluation():
    print("=" * 60)
    print("STEP 4a: Dynamic INT8 Quantization Evaluation")
    print("=" * 60)

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
    passages = sample_passages[:200]

    # Disable disk cache to measure pure inference
    settings.EMBED_CACHE_ENABLED = False

    # 1. Unquantized baseline
    settings.EMBED_QUANTIZE = False
    reset_model()
    t0 = time.perf_counter()
    embs_fp32 = embed_texts(passages, max_length=128)
    q_vec_fp32 = embed_query("What are the most effective techniques for optimizing large language model inference?", max_length=128)
    time_fp32 = time.perf_counter() - t0

    # Top-5 retrieval with FP32
    sims_fp32 = np.dot(embs_fp32, q_vec_fp32.squeeze())
    top5_fp32 = set(np.argsort(sims_fp32)[::-1][:5])

    # 2. Dynamic INT8 Quantized
    settings.EMBED_QUANTIZE = True
    reset_model()
    t0 = time.perf_counter()
    embs_int8 = embed_texts(passages, max_length=128)
    q_vec_int8 = embed_query("What are the most effective techniques for optimizing large language model inference?", max_length=128)
    time_int8 = time.perf_counter() - t0

    # Top-5 retrieval with INT8
    sims_int8 = np.dot(embs_int8, q_vec_int8.squeeze())
    top5_int8 = set(np.argsort(sims_int8)[::-1][:5])

    # Comparison metrics
    overlap = len(top5_fp32.intersection(top5_int8)) / 5.0
    cos_sims = np.sum(embs_fp32 * embs_int8, axis=1)
    mean_cos = float(np.mean(cos_sims))

    print(f"FP32 embedding time (200 texts): {time_fp32:.3f}s")
    print(f"INT8 embedding time (200 texts): {time_int8:.3f}s")
    print(f"Mean cosine similarity (FP32 vs INT8): {mean_cos:.6f}")
    print(f"Top-5 passage overlap: {overlap * 100:.1f}% ({len(top5_fp32.intersection(top5_int8))}/5)")

    # Reset
    settings.EMBED_QUANTIZE = False
    reset_model()
    return {
        "time_fp32": time_fp32,
        "time_int8": time_int8,
        "mean_cos": mean_cos,
        "top5_overlap": overlap,
    }


def run_openai_backend_evaluation():
    print("\n" + "=" * 60)
    print("STEP 4b: OpenAI Backend Evaluation & Cost Estimation")
    print("=" * 60)

    # Cost estimate for benchmark
    # 1567 passages * ~130 tokens = ~203,710 tokens
    # OpenAI text-embedding-3-small pricing: $0.02 / 1,000,000 tokens
    est_tokens = 1567 * 130
    est_cost = (est_tokens / 1_000_000) * 0.02
    print(f"Benchmark token estimate: ~{est_tokens:,} tokens across 1,567 passages")
    print(f"text-embedding-3-small cost estimate: ${est_cost:.5f} (approx $0.004)")

    # Test fallback behavior when OpenAI is unconfigured or fails
    orig_backend = settings.EMBEDDING_BACKEND
    settings.EMBEDDING_BACKEND = "openai"

    try:
        # If API key is present and configured, test real call; else verify fallback to local model
        res = embed_texts(["Test sentence for embedding."], max_length=128)
        print(f"Result shape: {res.shape}, L2-norm: {float(np.linalg.norm(res[0])):.4f}")
        print("Fallback / execution verified successfully without crash!")
    finally:
        settings.EMBEDDING_BACKEND = orig_backend


if __name__ == "__main__":
    run_quantization_evaluation()
    run_openai_backend_evaluation()
