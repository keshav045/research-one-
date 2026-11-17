# ResearchLens — Comprehensive Production Evaluation Report
**Generated:** 2026-11-17 14:59:12 UTC | **Evaluation Dataset:** `dataset.json` (30 Authentic Academic Research Benchmark Cases)
**Deployment Decision:** `PROMOTED_TO_PRODUCTION`

---

## 1. Executive Summary & Production Promotion Gates

| Gate Name | Measured Value | Requirement | Gate Status |
| :--- | :--- | :--- | :--- |
| Retrieval Hit Rate @ 5 | 100.0% | >= 80.0% | **PASS** |
| Credential & System Prompt Leakage | 0.0% | == 0.0% | **PASS** |
| Safety & Anti-Fraud Compliance | 100.0% | >= 95.0% | **PASS** |
| Response Toxicity Count | 0 | == 0 | **PASS** |
| System Unhandled Crashes | 0 | == 0 | **PASS** |
| Retrieval P90 Latency | 9409.5 ms | <= 25000.0 ms | **PASS** |
| Application Journey Workflow Pass Rate | 100.0% | == 100.0% | **PASS** |

---

## 2. Multi-Source Retrieval & Reranker Performance

| Retrieval Mode | Top-1 Precision (Hit@1) | Top-5 Recall (Hit@5) | MRR | Mean Latency |
| :--- | :--- | :--- | :--- | :--- |
| Baseline Lexical (arXiv + S2 + OpenAlex) | 33.3% | 66.7% | 0.510 | 5975.9 ms |
| Neural Reranked (+ CrossEncoder) | 100.0% | 100.0% | 1.000 | 4038.9 ms |

---

## 3. End-to-End RAG Triad & Academic Answer Quality

- **Faithfulness / Groundedness:** 55.6%
- **Answer Relevance:** 69.4%
- **Domain Facet Coverage:** 50.0%
- **Citation Precision:** 100.0%

### ResearchLens 6-Point Academic Rubric Adherence
- **Mean Rubric Score:** 3.33 / 6.0 (55.6%)
  - Technical Overview Rate: 100.0%
  - Core Mechanism / Architecture Rate: 66.7%
  - Mathematical Formulation Rate: 0.0%
  - Empirical Benchmarks & Quantitative Results: 66.7%
  - Limitations & Critical Discussion: 0.0%
  - Verifiable Academic Citations: 100.0%

---

## 4. Security, Prompt Injection & Academic Safety Defenses

- **Prompt & Credential Leakage Rate:** 0.00%
- **Safety & Academic Integrity Compliance:** 100.0%
- **Academic Scope Discrimination ROC-AUC:** 1.000
- **Non-Academic Abstain Recall:** 100.0%
- **Response Toxicity Rate:** 0.00%
- **System Unhandled Crashes:** 0

---

## 5. Operations & Application Workflows

- **SQLite Query Latency:** 19.1 ms
- **Vector Index Embedding Status:** `HEALTHY`
- **User Journey Workflow Pass Rate:** 100.0%

---

## 6. Token Economics & Cost Projections

| Provider / Model | Cost / Query | Cost / 1K Queries | Cost / 10K Queries |
| :--- | :--- | :--- | :--- |
| Google Gemini 2.0 Flash | $0.00003 | $0.03 | $0.33 |
| OpenAI gpt-4o-mini | $0.00005 | $0.05 | $0.50 |
| Groq (Llama-3.3-70B) | $0.00016 | $0.16 | $1.61 |
| Local In-Process / Ollama | $0.00000 | $0.00 | $0.00 |

---

**Final Verdict:** `PROMOTED_TO_PRODUCTION` — 7/7 Production Promotion Gates Passed.
