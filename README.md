# ResearchLens: Grounded Academic Synthesis Engine

> Production-grade, hallucination-resistant academic research assistant that pairs literature retrieval across arXiv, Semantic Scholar, and OpenAlex with verbatim PDF extraction, dual-gate atomic claim verification, answer-level NLI audit, and structured report synthesis.

---

## Architecture Overview

```
User Query
   │
   ▼
[Stage 1: Query Planner] ── Intent Classification (factual_lookup / literature_review) + Title Guesses
   │
   ▼
[Stage 2: Multi-Source Retrieval] ── Polite APIs: arXiv + Semantic Scholar (1.1s throttle) + OpenAlex
   │
   ▼
[Stage 3: Reranker & Anchor Ensemble] ── Citation Chasing ∩ Key-Phrase Gating ∩ Derivative Guard
   │
   ▼
[Stage 4: PDF Extraction Engine] ── PyMuPDF sentence windowing + 80MB cap + dynamic arXiv resolution
   │
   ▼
[Stage 5: Vector Indexing] ── FAISS cosine similarity / dense passage retrieval
   │
   ▼
[Stage 6: Dual-Gate Evidence & NLI] ── Atomic claim decomposition + DeBERTa-v3 cross-encoder
   │
   ▼
[Stage 7: Answer Writer & NLI Audit] ── 3-6 sentence synthesis + sentence-level NLI verification
   │
   ▼
[Verified Research Report] ── Plain-text answer, dynamic methodology, references, citation integrity %
```

---

## Tech Stack & Core Models

| Component | Technology / Model | Role |
|---|---|---|
| **Frontend** | React 19, TypeScript, Vite, TailwindCSS | Research workspace, PDF exporter, citation inspector |
| **Backend** | Python 3.11, FastAPI, SQLAlchemy, SQLite | Async orchestration, rate limiters, pipeline execution |
| **Passage Embeddings** | `BAAI/bge-small-en-v1.5` | Dense vector indexing across extracted passages |
| **Cross-Encoder Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Relevance scoring and candidate filtration |
| **NLI Verification** | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | Entailment attestation ($\ge 0.80$ threshold) |
| **Local LLM Synthesis** | `Qwen/Qwen2.5-0.5B-Instruct` or Ollama | Grounded answer generation using verified claims only |
| **PDF Extraction** | PyMuPDF (`fitz`) | Per-page parsing, header/footer/reference stripping |

---

## Quick Start & Setup

### Prerequisites
- **Python**: 3.10+ (tested on Python 3.11)
- **Node.js**: 20+ and **npm**
- **Hardware**: CUDA GPU recommended for fast DeBERTa-v3 NLI inference (automatically falls back to CPU)

### 1. Environment Configuration

Copy the example environment files for both backend and frontend:

```bash
# Root frontend configuration
cp .env.example .env

# Backend configuration
cp backend/.env.example backend/.env
```

Edit `backend/.env` with your API keys and model options (placeholders provided):

```ini
# --- LLM Provider ("local" | "ollama" | "qwen" | "gemini") ---
LLM_PROVIDER=local
LOCAL_LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct

# --- Scholarly APIs ---
SEMANTIC_SCHOLAR_API_KEY=your_s2_api_key_here
OPENALEX_EMAIL=your_email@domain.com
OPENALEX_API_KEY=

# --- Optional Cloud LLMs ---
GEMINI_API_KEY=your_gemini_api_key_here
QWEN_API_KEY=your_qwen_api_key_here

# --- Execution Flags ---
SEED_PAPERS_ENABLED=false
DISABLE_CACHE=false
```

### 2. Backend Installation

```bash
# From project root:
cd backend
python -m venv venv

# Windows:
venv\Scripts\activate
# macOS / Linux:
# source venv/bin/activate

pip install -r requirements.txt
cd ..
```

### 3. Frontend Installation

```bash
npm install
```

---

## Running the Application

Open two terminal windows:

### Terminal 1: Backend API Server
```bash
# Windows / Linux / macOS:
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
- API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health Check: [http://localhost:8000/health](http://localhost:8000/health)

### Terminal 2: Frontend Web App
```bash
npm run dev
```
- Web Application: [http://localhost:5173](http://localhost:5173)

---

## Running Tests

All unit, integration, and integrity tests are executed via `pytest`:

```bash
# Run Step 6 comprehensive test suite (NLI audit, status rules, false sentence rejection, ensemble)
pytest backend/tests/test_step6.py -v

# Run full test suite
pytest -v
```

---

## Key Pipeline Features

1. **Frozen Anchor Ensemble (`select_anchor_paper`)**:
   - Exact title in question / title-guess check.
   - Dual-engine arbitration: citation chasing ($f17f732$ reference-sharing) combined with key-phrase gating.
   - Divergence detection: flags `anchor_confidence = "uncertain"` and presents both foundational and variant candidates when heuristics differ.
   - Derivative guard: automatically detects variant prefixes ("3D", "Sentence-", "Group", "Fast", "Swin", "Rotary", "survey") and marks them uncertain.
2. **Hallucination-Proof Answer Synthesis (Step 6A)**:
   - Plain-text 3-6 sentence synthesis strictly constrained to verified claims with page-level attribution.
   - Compulsory first-sentence attribution naming authors, year, and paper title (never invented).
3. **Answer-Level NLI Verification (Step 6B)**:
   - Sentence-by-sentence entailment audit against source passages with 3-sentence context windows.
   - Sentences failing the $0.80$ entailment threshold are excised from the report.
   - Generates true Citation Integrity metrics ($\text{verified sentences} / \text{total sentences}$).
4. **Three Status Rules (Step 6C)**:
   - `completed`: Integrity $\ge 80\%$, anchor confidence high, full PDF text parsed.
   - `completed_with_warnings`: Integrity $< 80\%$, or anchor uncertain, or abstract-only fallback.
   - `insufficient_evidence`: Zero verified claims or zero entailed answer sentences (e.g. nonsense queries).

---

## Known Limitations

- **OpenAlex Rate Limits**: Unauthenticated calls share a 1,000 requests/day IP budget. Setting an email in `OPENALEX_EMAIL` activates polite pool usage, but high-volume benchmarks may exhaust the daily quota.
- **Semantic Scholar Throttling**: Without an API key, unauthenticated requests are limited to 1 call per 3 seconds and subject to frequent 429 rate limiting. Setting `SEMANTIC_SCHOLAR_API_KEY` enables polite 1.1s spacing and up to 6 searches per job.
- **Paywalled / Non-Open-Access PDFs**: Papers behind publisher paywalls (IEEE Xplore, ScienceDirect, Springer) without preprint mirrors on arXiv fall back to abstract-only analysis, which is flagged in the report limitations and triggers `completed_with_warnings`.
- **CPU Inference**: Running DeBERTa-v3 and Qwen2.5 locally on CPU without CUDA acceleration will result in longer per-question synthesis durations (~1-2 minutes per investigation).
