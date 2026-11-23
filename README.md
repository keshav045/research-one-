---
title: ResearchLens
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.35.0"
app_file: streamlit_app.py
pinned: false
---

# 🔬 ResearchLens

An open-source academic research assistant that actually reads the papers, verifies claims against real PDF text, and produces properly cited research reports without hallucinating.

---

## The Problem with LLMs in Academic Research

If you have ever asked ChatGPT, Claude, or any standard chatbot a research question like:
> *"Which paper introduced the Transformer architecture, and what was its BLEU score on WMT'14?"*

You've probably noticed that LLMs frequently:
- Cite papers that do not exist or invent non-existent DOIs.
- Attribute a technique to the wrong authors or the wrong year.
- Mix up numbers, benchmark results, and baseline models.
- Hallucinate citations (assigning `[1]` to a paper that was never mentioned).

**ResearchLens solves this.** Instead of asking a model to answer from memory, it searches real scholarly databases, downloads the actual PDFs, extracts the text, verifies every assertion using a Natural Language Inference (NLI) model, and generates an evidence-grounded summary where every single claim links back to a verified sentence in a real paper.

---

## How It Works: LangGraph State Machine with Adaptive Retry Loop

ResearchLens orchestrates its multi-stage academic pipeline as an explicit, stateful **LangGraph** state machine. Instead of a rigid linear pipeline that fails when a tricky query yields few papers, ResearchLens features an **autonomous self-correcting retrieval loop**:

```text
[ Your Research Question ]
           │
           ▼
    [ preflight ] ────(rejected: casual chat / invalid)────► [ END ]
           │
           ▼ (ok)
    [ plan_query ] (Classifies intent, decomposes sub-questions & search keywords)
           │
           ▼
    [ retrieve ] ◄──────────────────────────────────────────────────────┐
           │                                                            │
           ▼                                                            │
    [ rerank_fetch ] (MS-MARCO MiniLM reranking, PyMuPDF extraction)    │
           │                                                            │
           ▼                                                            │
    [ verify_claims ] (DeBERTa-v3 NLI entailment & source match)        │
           │                                                            │
    [ Enough verified claims? ]                                         │
      ├── No (and retries < 2) ──► [ rewrite_query ] ───────────────────┘
      │                            (LLM rewriter broadens keywords)
      │
      └── Yes (or retries exhausted)
           │
           ▼
    [ write_report ] (Academic synthesis, Reference [1] anchor locking)
           │
           ▼
        [ END ]
```

### Why LangGraph Instead of a Plain Function Chain?
1. **Explicit State & Conditional Cycles:** Academic research is non-linear. When initial candidate papers don't yield sufficient verifiable empirical facts, a plain linear chain fails with "no evidence". LangGraph makes the retry loop explicit, bounded (`MAX_RETRIES = 2`), and observable.
2. **Granular Telemetry & Streaming:** Each node returns only its modified state keys (`TypedDict`), enabling live step streaming into Streamlit and FastAPI with zero state corruption.
3. **Traceability & Testing:** Routing logic is isolated into pure conditional functions (`route_after_preflight`, `route_after_verify`) that can be tested in milliseconds without firing heavy neural models.

---

## Key Features

- **Autonomous Self-Correction (LangGraph):** Automatically reformulates search phrases and re-queries academic sources if initial candidate papers don't corroborate enough claims.
- **No Hallucinated Citations:** If a paper wasn't downloaded and verified, it doesn't get cited. Period.
- **Smart Anchor Paper Locking:** The main paper answering your question is locked as Reference `[1]`. Secondary or follow-up papers never hijack the primary reference spot.
- **Pre-Flight Query Guard:** Blocks spam, creative writing requests, and accidental paragraph pastes with friendly, clear guidance.
- **Multi-Source Coverage:** Gathers literature from arXiv, Semantic Scholar, and OpenAlex in parallel with automated exponential backoff on rate limits.
- **Full Text Provenance:** Tracks the exact page number and verbatim highlight sentence for every single verified claim.
- **Comparison Tables:** Automatically generates comparison matrices covering architecture, datasets, metrics, and key innovations.
- **LangSmith Tracing & Observability:** Live distributed tracing for every LangGraph node, query rewrite iteration, NLI verification verdict, and latency profile directly on [smith.langchain.com](https://smith.langchain.com).
- **Flexible LLMs:** Works out of the box with OpenAI (`gpt-4o-mini`, `gpt-4o`), Google Gemini (`gemini-2.0-flash`), Qwen, or local Hugging Face / Ollama models.
- **Cloud-Ready:** Automatically detects read-only environments (like Streamlit Community Cloud) and uses `/tmp` for SQLite and PDF caching with zero crashes.

---

## Quick Start (Get Running in 3 Minutes)

### 1. Prerequisites
- Python 3.11 or newer installed.
- (Optional) NVIDIA GPU with CUDA if you want fast local embeddings and NLI. If no GPU is available, it automatically falls back to CPU.

### 2. Clone and Setup Environment

```bash
# Clone the repository
git clone https://github.com/keshav045/research-one-.git
cd research-one-

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Create your `.env` File
Create a `.env` file in the project root:

```ini
# LLM Provider: openai | gemini | local
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-proj-your-openai-api-key-here
OPENAI_MODEL=gpt-4o-mini

# Optional: Google Gemini (if using LLM_PROVIDER=gemini)
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.0-flash

# Optional: Academic API keys (improves rate limits)
SEMANTIC_SCHOLAR_API_KEY=
OPENALEX_API_KEY=
OPENALEX_EMAIL=your_email@domain.com

# Models (defaults are fast and highly accurate)
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
NLI_MODEL=MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
NLI_DEVICE=cuda
NLI_ENTAIL_THRESHOLD=0.80

# LangSmith Tracing & Observability
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=researchlens

# Storage & Servers
DATABASE_URL=sqlite:///./researchlens.db
PDF_CACHE_DIR=./pdf_cache
HOST=127.0.0.1
PORT=8000
```

> **Tip:** If you are running on CPU without a GPU, simply change `NLI_DEVICE=cuda` to `NLI_DEVICE=cpu`.

### 4. Run the Project

You can run both the FastAPI backend and the Streamlit frontend with a single command:

```bash
# Launch both backend (port 8000) and frontend (port 8501)
python run_local.py
```

Or on Windows, double-click:
```bat
run_local.bat
```

Once running, open your browser:
- **Streamlit Web App:** [http://localhost:8501](http://localhost:8501)
- **FastAPI Backend:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## Deploying to Streamlit Cloud

If you want to host ResearchLens for free on Streamlit Community Cloud:

1. Fork or push this repository to your GitHub account.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and create a new app pointing to `streamlit_app.py`.
3. In **App settings > Secrets**, paste your configuration (matches the `.env` format):

```toml
LLM_PROVIDER = "openai"
OPENAI_API_KEY = "sk-proj-your-openai-api-key-here"
OPENAI_MODEL = "gpt-4o-mini"

# Optional academic keys
SEMANTIC_SCHOLAR_API_KEY = ""
OPENALEX_API_KEY = ""
OPENALEX_EMAIL = "your_email@domain.com"

# Models
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
NLI_DEVICE = "cpu"
SKIP_MODEL_WARMUP = "true"
MAX_PDF_WORKERS = "4"

# LangSmith Observability Tracing
LANGSMITH_TRACING = "true"
LANGSMITH_API_KEY = "your_langsmith_api_key"
LANGSMITH_PROJECT = "researchlens"
```

> **Note on Filesystems:** Streamlit Cloud has a read-only source directory. ResearchLens automatically detects this and stores SQLite databases in `/tmp/researchlens_st.db`, ensuring zero disk-permission errors.

---

## Project Structure

```text
research-one/
├── streamlit_app.py              # Main Streamlit user interface
├── run_local.py                  # One-click script launching backend & UI
├── run_local.bat                 # Windows one-click batch launcher
├── requirements.txt              # All dependencies (FastAPI, PyTorch, PyMuPDF, etc.)
├── requirements-streamlit.txt    # Streamlit Cloud deployment package list
│
├── backend/
│   ├── main.py                   # FastAPI app entry point & lifespan events
│   ├── config.py                 # Pydantic Settings & environment variables
│   ├── security.py               # Pre-flight query validation & narrative filters
│   ├── models/
│   │   ├── database.py           # SQLAlchemy SQLite setup & cloud /tmp fallback
│   │   └── schemas.py            # Pydantic schemas (Paper, Citation, Report)
│   ├── routers/
│   │   └── research.py           # REST endpoints (/api/research/*)
│   ├── services/
│   │   ├── paper_search_manager.py # Orchestrates parallel retrieval across sources
│   │   ├── paper_retrieval.py      # arXiv, Semantic Scholar & OpenAlex API clients
│   │   ├── pdf_extractor.py        # PyMuPDF engine, magic bytes & watermark removal
│   │   ├── ranker.py               # Canonical deduplication & MS-MARCO reranking
│   │   ├── vector_store.py         # Paper-scoped embedding indices (BGE-small)
│   │   ├── evidence.py             # Verbatim sentence extraction & claim matching
│   │   ├── nli_verifier.py         # DeBERTa-v3 Natural Language Inference audit
│   │   ├── local_llm_service.py    # Report synthesis, citation badge verification
│   │   └── query_planner.py        # Question intent classifier & sub-query builder
│   └── tests/                    # Pytest test suite
│
└── evaluation/
    ├── benchmark.py              # Ground-truth evaluation runner
    ├── harness.py                # Dataset loader and test lifecycle harness
    ├── metrics.py                # Recall@k, MRR, NDCG@10, and Citation Precision
    ├── run_suite.py              # 13-module master evaluation suite
    └── dataset.json              # Curated academic research benchmark questions
```

---

## Running Tests & Benchmarks

ResearchLens includes a full test suite to guarantee citation boundaries, deduplication, and query safety:

```bash
# Run unit and integration tests
pytest backend/tests -v

# Run the evaluation benchmark on 5 sample queries
python -m evaluation.benchmark --max 5

# Run benchmark filtered by difficulty tier
python -m evaluation.benchmark --difficulty easy --max 3
python -m evaluation.benchmark --difficulty medium --max 3
python -m evaluation.benchmark --difficulty hard --max 3
```

---

## Frequently Asked Questions (FAQ)

#### 1. What happens if Semantic Scholar hits a rate limit?
Semantic Scholar has strict rate limits for free tiers. ResearchLens catches HTTP 429 errors automatically, applies exponential backoff, and ensures arXiv and OpenAlex results are still returned without failing your research request.

#### 2. Can I run this completely offline or locally?
Yes! Set `LLM_PROVIDER=local` in `.env` and set `NLI_DEVICE=cpu`. The embedding model, reranker, and NLI model will all run locally on your machine via Hugging Face Transformers.

#### 3. Why was my question rejected?
ResearchLens enforces research-focused queries between 10 and 500 characters. If you paste a random story or ask a generic question like *"write a poem"* or *"tell me a joke"*, the pre-flight guard will politely prompt you to ask an academic or scientific research question.

---

## 💼 LangGraph Engineering & Interview Talking Points

### Resume Bullet
> *Orchestrated the academic synthesis pipeline as a LangGraph state machine with conditional routing and an autonomous retrieval-retry loop triggered when NLI-verified empirical claims fall below threshold; improved evidence corroboration and prevented synthesis hallucinations across multi-hop research queries.*

### Interview Q&A
- **Why LangGraph instead of a plain linear chain?**
  A linear chain is brittle: if initial keyword searches miss relevant preprints, the entire pipeline fails with an "insufficient evidence" error. LangGraph enables stateful cycles and conditional edges, turning retrieval into an adaptive, self-correcting loop.
- **What is in the graph state, and what does each node return?**
  State is a typed `ResearchState` (`TypedDict`) containing serializable fields (`question`, `sub_queries`, `papers`, `passages`, `verified`, `rejected`, `retries`). Each node strictly returns only the keys it updates, avoiding data pollution or unintended state overwrites.
- **How do you prevent infinite retry loops?**
  A strict `MAX_RETRIES = 2` guard is enforced by the conditional routing function (`route_after_verify`). LangGraph's internal recursion limit acts as an auxiliary safeguard, but application-level state bounds guarantee deterministic termination.
- **What happens if retries are exhausted without enough evidence?**
  The graph routes to `write_report`, which produces an honest, transparent report marked as **"Limited Evidence"**, clearly documenting which claims could not be corroborated rather than hallucinating facts.
- **What would you improve next?**
  Adding checkpointing (e.g., `MemorySaver` or SQLite checkpointer) for resumable investigations, and an optional human-in-the-loop review node when a report triggers the limited-evidence warning.

---

## License

This project is open-source under the [MIT License](LICENSE). Contributions, bug reports, and pull requests are warmly welcome!
