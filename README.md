# ResearchLens

> AI-powered academic research synthesis with real paper retrieval, PDF extraction, NLI-verified citations, and Gemini report generation.

## Tech Stack

| Component | Technology |
|---|---|
| Frontend | React 19 + TypeScript + Vite |
| Backend | Python 3.11 + FastAPI |
| Paper retrieval | arXiv API + Semantic Scholar API |
| PDF extraction | PyMuPDF |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector database | FAISS |
| Claim verification | Hugging Face NLI (`cross-encoder/nli-deberta-v3-small`) |
| LLM synthesis | Gemini API (`gemini-2.0-flash`) |
| Database | SQLite → PostgreSQL |

---

## Quick Start

### Prerequisites
- **Node.js** 20+ and **npm**
- **Python** 3.10+

### 1. Clone & configure

```bash
git clone <your-repo-url>
cd researchlens

# Frontend env
cp .env.example .env
# Edit .env → set VITE_GEMINI_API_KEY and VITE_API_URL

# Backend env
cp backend/.env.example backend/.env
# Edit backend/.env → set GEMINI_API_KEY
```

### 2. Install dependencies

```bash
# Frontend
npm install

# Backend (creates venv + installs all Python packages)
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS
pip install -r requirements.txt
cd ..
```

### 3. Run (two terminals)

**Terminal 1 — Backend:**
```bash
npm run backend:dev
# Server starts at http://localhost:8000
# Docs at http://localhost:8000/docs
```

**Terminal 2 — Frontend:**
```bash
npm run dev
# App starts at http://localhost:5173
```

---

## Production Build

### Frontend
```bash
npm run build          # outputs to dist/
npm run preview        # serve the production build locally
```

### Backend
```bash
# From project root, using production-grade server:
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4
```

> **Note:** On first run the backend downloads two HuggingFace models (~370 MB total, cached permanently):
> - `all-MiniLM-L6-v2` — embeddings
> - `cross-encoder/nli-deberta-v3-small` — NLI claim verification

---

## Deployment

### Option A — Vercel (Frontend) + Railway / Render (Backend)

1. Push to GitHub
2. Connect repo to Vercel → set `VITE_GEMINI_API_KEY` and `VITE_API_URL` env vars
3. Deploy backend to Railway/Render → set `GEMINI_API_KEY`
4. Update `VITE_API_URL` in Vercel to point at the deployed backend URL

### Option B — Docker (Full stack)

```bash
# Build
docker compose up --build

# Separate services
docker compose up frontend backend
```

### Option C — Single VPS

```bash
# Build frontend static files
npm run build

# Serve frontend via Nginx, proxy /api → uvicorn
# Run backend: uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

---

## API Reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Backend health + config check |
| `POST` | `/api/research` | Start a new research job |
| `GET` | `/api/research` | List all jobs |
| `GET` | `/api/research/{id}` | Poll job status / get report |
| `DELETE` | `/api/research/{id}` | Delete a job |
| `GET` | `/api/research/{id}/papers` | Papers retrieved for a job |

Interactive docs: **http://localhost:8000/docs**

---

## Environment Variables

### Frontend (`.env`)
| Variable | Description |
|---|---|
| `VITE_GEMINI_API_KEY` | Gemini API key for browser-side fallback |
| `VITE_API_URL` | Python backend URL (e.g. `http://localhost:8000`) |

### Backend (`backend/.env`)
| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Gemini API key |
| `GEMINI_MODEL` | Model name (default: `gemini-2.0-flash`) |
| `SEMANTIC_SCHOLAR_API_KEY` | Optional — higher rate limits |
| `DATABASE_URL` | SQLite (default) or PostgreSQL connection string |
| `NLI_MODEL` | HuggingFace NLI model name |
| `EMBEDDING_MODEL` | Sentence Transformer model name |
| `PDF_CACHE_DIR` | Directory for cached PDFs |
