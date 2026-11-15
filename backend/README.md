# ResearchLens Backend

Full Python backend implementing the complete research synthesis pipeline.

## Tech Stack

| Component | Technology |
|---|---|
| Frontend | React + TypeScript (Vite) |
| **Backend** | **Python + FastAPI** |
| **Workflow** | **Python async (LangGraph-ready)** |
| **Paper retrieval** | **arXiv API + Semantic Scholar API** |
| **PDF extraction** | **PyMuPDF (fitz)** |
| **Embeddings** | **Sentence Transformers (all-MiniLM-L6-v2)** |
| **Vector database** | **FAISS (faiss-cpu)** |
| **Claim verification** | **Hugging Face NLI (cross-encoder/nli-deberta-v3-small)** |
| **LLM report generation** | **Gemini API** |
| **Database** | **SQLite → PostgreSQL** |

## Setup

### 1. Create & activate a virtual environment

```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note**: First install downloads ~2 GB (PyTorch + Transformers + Sentence Transformers). Subsequent starts use the cache.

### 3. Configure environment

```bash
copy .env.example .env
# Edit .env and fill in:
#   GEMINI_API_KEY=your_key_here
```

### 4. Start the backend server

```bash
# From the PROJECT ROOT (not the backend folder):
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Or use the convenience script:

```bash
backend\start.bat              # Windows
```

### 5. Start the frontend (separate terminal)

```bash
npm run dev
```

The frontend `.env` has `VITE_API_URL=http://localhost:8000` which points it at the backend.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/research` | Start a new research job |
| `GET` | `/api/research` | List all jobs |
| `GET` | `/api/research/{id}` | Poll job status / get report |
| `DELETE` | `/api/research/{id}` | Delete a job |
| `GET` | `/api/research/{id}/papers` | Get retrieved papers for a job |
| `GET` | `/health` | Health check |

## Pipeline Steps

```
1. Question Analysis      → keyword extraction, query planning
2. Research Planning      → arXiv & Semantic Scholar query construction
3. Paper Retrieval        → real paper metadata + PDF URLs
4. PDF Extraction         → PyMuPDF full-text, page-by-page passages
5. Embedding & Indexing   → Sentence Transformer embeddings → FAISS
6. Report Generation      → Gemini API (grounded in real paper metadata)
7. Claim Verification     → HuggingFace NLI entailment scoring per atomic claim
```

## Model Downloads (first run only)

| Model | Size | Purpose |
|---|---|---|
| `all-MiniLM-L6-v2` | ~90 MB | Semantic embeddings |
| `cross-encoder/nli-deberta-v3-small` | ~280 MB | NLI claim verification |

Models are cached in `~/.cache/huggingface/hub/` after first download.

## Upgrading to PostgreSQL

Change `DATABASE_URL` in `.env`:

```
DATABASE_URL=postgresql+psycopg2://user:password@localhost/researchlens
```

Then: `pip install psycopg2-binary`
