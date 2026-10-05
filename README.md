# ResearchLens

ResearchLens is an autonomous academic research assistant that searches open-access literature, extracts full text from PDFs, validates factual assertions using natural language inference (NLI), and synthesizes cited research reports.

The engine prioritizes evidence integrity over generated prose. Every factual claim in the final report must be corroborated by verbatim passages retrieved from actual papers. Uncorroborated assertions and hallucinated references are filtered out before reports are finalized.

---

## How It Works

1. **Query Planning**: The system analyzes the research question, detects whether the query is a factual lookup or a broader literature review, and decomposes it into targeted sub-queries.
2. **Multi-Source Retrieval**: Queries are dispatched concurrently to arXiv, Semantic Scholar, and OpenAlex.
3. **Canonical Deduplication**: Overlapping records across sources are merged deterministically using a strict priority cascade (DOI, arXiv ID, Semantic Scholar ID, OpenAlex ID, and normalized title).
4. **Relevance Filtering**: Candidates are scored and filtered by token overlap and semantic relevance. Irrelevant records are excluded before downloading PDFs.
5. **PDF Extraction**: The engine retrieves full-text PDFs, validates PDF magic bytes, strips headers, footers, and references, and chunks the remaining text into windowed passages.
6. **Paper-Scoped Vector Search**: Extracted passages are embedded and indexed. Passage retrieval is strictly scoped to the paper being evaluated, preventing cross-paper vector contamination.
7. **NLI Claim Verification**: Candidate factual assertions are cross-checked against verbatim passages using a fine-tuned DeBERTa-v3 cross-encoder model.
8. **Citation Audit & Synthesis**: Verified findings are synthesized with bracketed citations (`[1]`, `[2]`). Citation badge numbers are audited against the reference list, and any invalid references are removed.
9. **Transparent Metrics**: The final report includes a comparison table, citation integrity percentages, evidence coverage ratios, and an overall confidence level (High, Medium, or Low).

---

## Tech Stack

- **Frontend**: React 19, TypeScript, Vite, Tailwind CSS
- **Backend**: Python 3.11, FastAPI, SQLAlchemy, SQLite
- **PDF Extraction**: PyMuPDF (`fitz`)
- **Embeddings**: Sentence Transformers (`BAAI/bge-small-en-v1.5`)
- **Claim Verification**: Hugging Face DeBERTa-v3 NLI (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`)
- **Synthesis LLM**: Google Gemini 2.0 Flash (cloud) or local Ollama / Hugging Face models

---

## Project Structure

```
├── backend/
│   ├── main.py                     # FastAPI application and lifespan management
│   ├── config.py                   # Environment settings and threshold constants
│   ├── models/
│   │   ├── database.py             # SQLAlchemy models and SQLite connection
│   │   └── schemas.py              # Pydantic data schemas
│   ├── routers/
│   │   └── research.py             # REST API endpoints (/api/research)
│   ├── services/
│   │   ├── paper_search_manager.py # Multi-source retrieval coordinator
│   │   ├── paper_retrieval.py      # arXiv, Semantic Scholar, and OpenAlex connectors
│   │   ├── pdf_extractor.py        # PDF downloader, magic-byte checker, text windowing
│   │   ├── ranker.py               # Canonical deduplication and relevance scoring
│   │   ├── vector_store.py         # Paper-scoped passage retrieval
│   │   ├── evidence.py             # Claim extraction and source-passage matching
│   │   ├── nli_verifier.py         # DeBERTa-v3 entailment auditing
│   │   ├── local_llm_service.py    # Report synthesis and citation validation
│   │   └── query_planner.py        # Question classification and sub-query generation
│   └── tests/                      # Pytest test suite
├── src/                            # React 19 frontend application
│   ├── components/                 # UI panels (report, papers, evidence, progress)
│   ├── services/api.ts             # API client connecting to backend
│   └── types/index.ts              # TypeScript interface definitions
├── public/                         # Static assets and Netlify redirect rules
├── scripts/                        # Benchmark and validation scripts
├── netlify.toml                    # Netlify frontend build and routing configuration
└── render.yaml                     # Render backend blueprint specification
```

---

## Local Development

### Prerequisites

- Python 3.11
- Node.js 20+ and npm

### 1. Backend Setup

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
# source venv/bin/activate

pip install -r requirements.txt
```

Create a `.env` file in the `backend/` directory:

```ini
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.0-flash

# Optional scholarly API keys
SEMANTIC_SCHOLAR_API_KEY=
OPENALEX_API_KEY=
OPENALEX_EMAIL=your_email@example.com

# Server configuration
HOST=0.0.0.0
PORT=8000
DATABASE_URL=sqlite:///./researchlens.db
PDF_CACHE_DIR=./pdf_cache
```

Start the backend:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`. You can check the health status at `http://localhost:8000/health` and browse the interactive docs at `http://localhost:8000/docs`.

### 2. Frontend Setup

From the root project directory:

```bash
npm install
npm run dev
```

The frontend will run at `http://localhost:5173`.

---

## Running Tests

The test suite covers unit tests, pipeline stages, citation boundaries, and paper-scoped retrieval:

```bash
# Run backend tests
pytest backend/tests

# Run frontend typecheck
npx tsc --noEmit

# Test production frontend build
npm run build
```

---

## Deployment

### Backend on Render

1. Create a new Web Service on Render from your Git repository.
2. Set the following build and start commands:
   - **Environment**: Python 3
   - **Build Command**:
     ```bash
     pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r backend/requirements.txt
     ```
   - **Start Command**:
     ```bash
     uvicorn backend.main:app --host 0.0.0.0 --port $PORT
     ```
3. Set these environment variables in Render:
   - `PYTHON_VERSION`: `3.11.9`
   - `HOST`: `0.0.0.0`
   - `LLM_PROVIDER`: `gemini`
   - `GEMINI_API_KEY`: your Gemini API key
   - `GEMINI_MODEL`: `gemini-2.0-flash`
   - `NLI_DEVICE`: `cpu`
   - `CORS_ORIGINS`: `*`
   - `OPENALEX_EMAIL`: your email address
   - `SKIP_MODEL_WARMUP`: `true`
4. Once deployed, note your service URL (for example, `https://researchlens-backend.onrender.com`).

### Frontend on Netlify

1. Connect your repository to Netlify.
2. Netlify reads the build settings from `netlify.toml` automatically:
   - **Build command**: `npm run build`
   - **Publish directory**: `dist`
3. Add one environment variable in Netlify Site Configuration:
   - `VITE_API_URL`: your Render backend URL (for example, `https://researchlens-backend.onrender.com`, without trailing slash)
4. Deploy the site.

---

## License

This project is licensed under the MIT License.
