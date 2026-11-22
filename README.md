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

## How It Works (Under the Hood)

Here is what happens when you enter a research question:

```
[ Your Research Question ]
           │
           ▼
1. Pre-Flight Check & Query Planner
   - Rejects non-research queries (e.g. casual chat, code snippets, random text pastes)
   - Checks length (10 to 500 characters)
   - Classifies query (Factual Lookup vs. Comparative Review)
   - Generates targeted search phrases
           │
           ▼
2. Multi-Source Retrieval
   - Searches arXiv, Semantic Scholar, and OpenAlex concurrently
   - Merges duplicates using DOI, arXiv ID, and title matching
   - Preserves citation counts and open-access PDF links
           │
           ▼
3. Relevance Filtering & Reranking
   - Uses cross-encoder rerankers (MS-MARCO MiniLM) to score candidates
   - Disqualifies off-topic papers (even if they have 50,000+ citations)
   - Selects the top relevant papers for full-text analysis
           │
           ▼
4. PDF Fetching & Cleanup
   - Validates PDF magic bytes (%PDF-) to ignore broken HTML redirects
   - Strips running headers, footers, and arXiv margin stamps (e.g. "arXiv:2103.14030v2")
   - Chunks text into clean, windowed technical passages
           │
           ▼
5. Paper-Scoped Vector Search
   - Embeds passages using BAAI/bge-small-en-v1.5
   - Scopes vector search strictly within each paper (zero cross-paper pollution)
           │
           ▼
6. Two-Tier Claim Verification
   - Tier 1: Checks for verbatim matching sentences in the source text
   - Tier 2: Runs DeBERTa-v3 NLI to verify that the passage strictly entails the claim
   - Discards any claim that doesn't meet the entailment threshold (≥ 0.80)
           │
           ▼
7. Report Synthesis & Citation Locking
   - Locks the primary foundational paper at Reference [1]
   - Audits all [n] badges to make sure they match the reference list
   - Builds a side-by-side comparison matrix and computes citation integrity scores
           │
           ▼
[ Final Verified Academic Report ]
```

---

## Key Features

- **No Hallucinated Citations:** If a paper wasn't downloaded and verified, it doesn't get cited. Period.
- **Smart Anchor Paper Locking:** The main paper answering your question is locked as Reference `[1]`. Secondary or follow-up papers never hijack the primary reference spot.
- **Pre-Flight Query Guard:** Blocks spam, creative writing requests, and accidental paragraph pastes with friendly, clear guidance.
- **Multi-Source Coverage:** Gathers literature from arXiv, Semantic Scholar, and OpenAlex in parallel with automated exponential backoff on rate limits.
- **Full Text Provenance:** Tracks the exact page number and verbatim highlight sentence for every single verified claim.
- **Comparison Tables:** Automatically generates comparison matrices covering architecture, datasets, metrics, and key innovations.
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

## License

This project is open-source under the [MIT License](LICENSE). Contributions, bug reports, and pull requests are warmly welcome!
