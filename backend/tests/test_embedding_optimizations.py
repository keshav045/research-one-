"""
Unit tests for passage embedding and vector indexing optimizations:
1. Order restoration after length-sorting
2. Cache hit/miss correctness
3. Skipping non-body sections
4. Near-duplicate removal
5. Query LRU cache
6. Embeddings normalized (norm ~ 1)
"""

import numpy as np

from backend.config import settings
from backend.models.schemas import Paper, PaperPassage
from backend.services.embeddings import (
    clear_disk_cache,
    clear_query_cache,
    embed_query,
    embed_texts,
    _lookup_disk_cache,
    _compute_cache_key,
)
from backend.services.vector_store import VectorStore


def test_order_restoration_after_length_sorting():
    """Verify that length-sorting in embed_texts restores exact input order."""
    orig_cache = settings.EMBED_CACHE_ENABLED
    settings.EMBED_CACHE_ENABLED = False
    try:
        texts = [
            "A short sentence.",
            "This is a much longer sentence designed to have significantly more tokens than the first one.",
            "Medium length text with some academic words like quantization.",
            "Tiny.",
            "Another extended passage with multiple clauses explaining transformer inference optimization and memory reduction.",
        ]

        # Embed as batch (which uses length sorting internally)
        batch_embs = embed_texts(texts, max_length=128)
        assert batch_embs.shape[0] == len(texts)

        # Embed individually to get un-permuted baseline
        single_embs = np.vstack([embed_texts([t], max_length=128) for t in texts])

        # Verify each position matches
        for i in range(len(texts)):
            cos_sim = float(np.dot(batch_embs[i], single_embs[i]))
            assert cos_sim > 0.999, f"Order mismatch at index {i}: cos_sim={cos_sim}"
    finally:
        settings.EMBED_CACHE_ENABLED = orig_cache


def test_cache_hit_miss_correctness():
    """Verify SQLite disk cache records misses, saves them, and returns hits on subsequent calls."""
    orig_cache = settings.EMBED_CACHE_ENABLED
    settings.EMBED_CACHE_ENABLED = True
    clear_disk_cache()
    try:
        texts1 = ["Passage alpha for testing disk cache.", "Passage beta for testing disk cache."]
        model_name = settings.EMBEDDING_MODEL
        max_len = 128
        keys1 = [_compute_cache_key(model_name, max_len, t) for t in texts1]

        # First call: cache should be empty
        init_lookup = _lookup_disk_cache(keys1)
        assert len(init_lookup) == 0

        embs1 = embed_texts(texts1, max_length=max_len)
        assert embs1.shape[0] == 2

        # Second lookup: keys should now be in disk cache
        after_lookup = _lookup_disk_cache(keys1)
        assert len(after_lookup) == 2
        for k in keys1:
            assert np.allclose(after_lookup[k], embs1[keys1.index(k)], atol=1e-5)

        # Mixed call: texts1 (hits) + texts2 (misses)
        texts2 = ["Passage gamma newly added to the batch."]
        all_texts = texts1 + texts2
        combined_embs = embed_texts(all_texts, max_length=max_len)
        assert combined_embs.shape[0] == 3

        # Check combined output matches previously cached for texts1
        assert np.allclose(combined_embs[0], embs1[0], atol=1e-5)
        assert np.allclose(combined_embs[1], embs1[1], atol=1e-5)
    finally:
        clear_disk_cache()
        settings.EMBED_CACHE_ENABLED = orig_cache


def test_skipping_nonbody_sections():
    """Verify SKIP_NONBODY_SECTIONS drops References, Bibliography, Appendix, Acknowledgements."""
    orig_skip = settings.SKIP_NONBODY_SECTIONS
    settings.SKIP_NONBODY_SECTIONS = True
    try:
        paper = Paper(
            id="test-paper-sections",
            title="Efficient Attention Mechanics",
            authors=["Alice"],
            publicationYear=2024,
            journalConference="NeurIPS",
            doi="10.1234/test-sections",
            abstract="Test abstract for paper with multiple sections.",
            source="arXiv",
            passages=[
                PaperPassage(id="p1", paper_id="test-paper-sections", page=1, section="Introduction", text="Introduction passage describing core motivation."),
                PaperPassage(id="p2", paper_id="test-paper-sections", page=2, section="Methodology", text="Methodology passage detailing the KV-cache algorithm."),
                PaperPassage(id="p3", paper_id="test-paper-sections", page=8, section="References", text="[1] Vaswani et al. Attention is all you need."),
                PaperPassage(id="p4", paper_id="test-paper-sections", page=9, section="Bibliography", text="Further citations and reference details."),
                PaperPassage(id="p5", paper_id="test-paper-sections", page=9, section="Acknowledgements", text="We thank the grant funding agency for compute support."),
                PaperPassage(id="p6", paper_id="test-paper-sections", page=10, section="Appendix", text="Supplementary proofs and derivations."),
            ],
        )

        vs = VectorStore()
        vs.build([paper])

        indexed_sections = {r.passage.section for r in vs.records}
        assert "Introduction" in indexed_sections
        assert "Methodology" in indexed_sections
        assert "References" not in indexed_sections
        assert "Bibliography" not in indexed_sections
        assert "Acknowledgements" not in indexed_sections
        assert "Appendix" not in indexed_sections
        assert len(vs.records) == 2
    finally:
        settings.SKIP_NONBODY_SECTIONS = orig_skip


def test_near_duplicate_removal():
    """Verify duplicate and near-duplicate passages within the same paper are dropped."""
    paper = Paper(
        id="test-paper-dup",
        title="Deduplication Benchmark Paper",
        authors=["Bob"],
        publicationYear=2024,
        journalConference="ICLR",
        doi="10.1234/test-dup",
        abstract="Test abstract for duplicate detection benchmark.",
        source="arXiv",
        passages=[
            PaperPassage(id="p1", paper_id="test-paper-dup", page=1, section="Method", text="PagedAttention partitions the key-value cache into non-contiguous blocks."),
            # Near-duplicate: casing, extra spaces, trailing punctuation
            PaperPassage(id="p2", paper_id="test-paper-dup", page=1, section="Method", text="  Pagedattention partitions the key-value cache into non-contiguous blocks... "),
            PaperPassage(id="p3", paper_id="test-paper-dup", page=2, section="Experiments", text="Evaluation shows significant latency reduction on A100 GPUs."),
        ],
    )

    vs = VectorStore()
    vs.build([paper])

    assert len(vs.records) == 2
    assert vs.records[0].passage.id == "p1"
    assert vs.records[1].passage.id == "p3"


def test_query_lru_cache():
    """Verify embed_query caches results and handles eviction/cache hits."""
    clear_query_cache()
    query = "What is FlashAttention?"

    emb1 = embed_query(query)
    emb2 = embed_query(query)

    assert isinstance(emb1, np.ndarray)
    assert emb1.shape == emb2.shape
    assert np.allclose(emb1, emb2, atol=1e-6)

    # Mutating returned copy should not affect cache
    emb1[0, 0] = 999.0
    emb3 = embed_query(query)
    assert not np.isclose(emb3[0, 0], 999.0)


def test_embeddings_normalized():
    """Verify all produced embeddings have L2 norm approximately equal to 1.0."""
    texts = [
        "First test sentence for normalization check.",
        "Second text verifying dense vector normalization.",
        "Third query sentence.",
    ]
    embs = embed_texts(texts, max_length=128)
    norms = np.linalg.norm(embs, axis=1)

    assert np.allclose(norms, 1.0, atol=1e-4)

    q_emb = embed_query("Query normalization check")
    q_norm = float(np.linalg.norm(q_emb))
    assert np.isclose(q_norm, 1.0, atol=1e-4)
