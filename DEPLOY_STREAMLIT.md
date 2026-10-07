# ResearchLens — Streamlit Community Cloud Deployment Guide

This guide explains how to deploy the standalone Streamlit edition of **ResearchLens** on [Streamlit Community Cloud](https://share.streamlit.io) without disrupting the existing FastAPI and React architectures.

---

## Architecture Overview

```
                      Streamlit Community Cloud
 ┌──────────────────────────────────────────────────────────────────┐
 │                                                                  │
 │   streamlit_app.py (Single-Process Self-Contained App)           │
 │    ├── UI & Live Progress (st.status / st.progress / Tabs)       │
 │    ├── In-Process Async Pipeline (run_research_pipeline)         │
 │    ├── Cached ML Models (@st.cache_resource on CPU):             │
 │    │    ├── Sentence Transformers: BAAI/bge-small-en-v1.5        │
 │    │    ├── Cross-Encoder Reranker: ms-marco-MiniLM-L-6-v2       │
 │    │    └── NLI Claim Verifier: nli-deberta-v3-small             │
 │    ├── Local SQLite DB (researchlens_st.db)                      │
 │    └── Cloud LLM Synthesis: Gemini 2.0 Flash / OpenAI gpt-4o-mini│
 │                                                                  │
 └────────────────────────────────┬─────────────────────────────────┘
                                  │ HTTPS Requests
                                  ▼
                [ arXiv / Semantic Scholar / OpenAlex APIs ]
```

---

## Step 1: Push Repository to GitHub

Ensure your latest changes are pushed to your GitHub repository:

```bash
git add streamlit_app.py requirements.txt requirements-streamlit.txt packages.txt .streamlit/ DEPLOY_STREAMLIT.md .gitignore
git commit -m "Add Streamlit Cloud standalone application"
git push origin main
```

> [!IMPORTANT]
> Verify that `.streamlit/secrets.toml` is **not** staged or committed. It is protected in `.gitignore`.

---

## Step 2: Create App on Streamlit Community Cloud

1. Log into [share.streamlit.io](https://share.streamlit.io) with your GitHub account.
2. Click **New app**.
3. Choose your repository:
   - **Repository**: `<your-github-username>/<your-repo-name>`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
   - **App URL**: Choose a custom subdomain (e.g. `researchlens-ai.streamlit.app`)

---

## Step 3: Configure App Secrets

Before deploying, click **Advanced settings...** (or go to **App Settings $\rightarrow$ Secrets**):

Paste your configuration (refer to [`.streamlit/secrets.toml.example`](./.streamlit/secrets.toml.example)):

```toml
# Primary Cloud LLM Provider
LLM_PROVIDER = "gemini"

# Google Gemini API Key (Recommended for high free-tier rate limits)
GEMINI_API_KEY = "AIzaSy..."
GEMINI_MODEL = "gemini-2.0-flash"

# Optional: OpenAI (if choosing LLM_PROVIDER = "openai")
# OPENAI_API_KEY = "sk-..."
# OPENAI_MODEL = "gpt-4o-mini"

# Optional: Academic APIs
# SEMANTIC_SCHOLAR_API_KEY = "s2k-..."
# OPENALEX_API_KEY = "..."
OPENALEX_EMAIL = "your_email@example.com"

# Memory & CPU Constraints for Streamlit Community Cloud (2.7 GB RAM limit)
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
NLI_DEVICE = "cpu"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
SKIP_MODEL_WARMUP = "true"
MAX_PDF_WORKERS = "2"
```

Click **Save** and then **Deploy!**

---

## Step 4: Verify Deployment

1. The first build installs requirements from `requirements.txt` and packages from `packages.txt`.
2. When the app boots:
   - Cold start time: **30–50 seconds** during the first run as Hugging Face downloads lightweight model weights (`bge-small` ~90MB, `nli-deberta-v3-small` ~280MB).
   - Once downloaded, models are cached in `@st.cache_resource` for instant reuse.
3. Test a query:
   > *"Which paper introduced the Transformer architecture, and what was its key idea?"*
4. Confirm:
   - Progress transitions across all 7 pipeline stages.
   - Status resolves to `Completed` with confidence score.
   - Per-sentence verification badges (`ENTAILS`, `NEUTRAL`) display correctly.
   - Report Markdown download works.

---

## Community Cloud Limits & Best Practices

| Resource | Constraint | Mitigation in ResearchLens |
| :--- | :--- | :--- |
| **Memory (RAM)** | ~2.7 GB max | Uses `bge-small`, `nli-deberta-v3-small`, PyTorch CPU single thread, and default `Quick` depth (3–4 papers). |
| **vCPU** | ~2 shared cores | Lazy model loading via `@st.cache_resource`; PDF workers throttled to 2. |
| **Concurrency** | Single container | Guarded with `threading.Lock` so only one pipeline run executes in memory at a time. |
| **Filesystem & Disk Cache** | Ephemeral | SQLite DB, local disk response cache, and passage embedding cache (`embedding_cache.sqlite`) are lost when the app container restarts or hibernates. UI includes clear notices. |
| **Outbound IP & API Limits** | Shared Cloud IP | On Streamlit Community Cloud the egress IP is shared across apps, so unauthenticated Semantic Scholar and arXiv calls can be rate-limited quickly. Always configure `SEMANTIC_SCHOLAR_API_KEY` and `OPENALEX_EMAIL` in Streamlit secrets for stable throughput. |
| **Cold Starts** | Apps hibernate on idle | `SKIP_MODEL_WARMUP=true` allows instant app UI render; models load only when research begins. |

---

## Architectural Fallback: Thin Streamlit Client

If your research queries frequently exceed Community Cloud's 2.7 GB RAM limit (e.g., analyzing 25+ preprints in `Deep` mode):

You can configure Streamlit as a **thin client** that submits queries to your dedicated FastAPI backend:
1. Deploy the FastAPI backend using Docker or a cloud container provider.
2. Point Streamlit to `https://<your-backend-host>/api/research` via HTTP POST and poll the job endpoint.
3. This completely frees Streamlit container memory from PyTorch and DeBERTa model overhead.
