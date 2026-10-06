# ResearchLens Deployment Guide: Netlify (Frontend) & Render (Backend)

This guide walks you through deploying **ResearchLens** with the **FastAPI backend on Render** and the **React + Vite frontend on Netlify**.

---

## Architecture Overview

```
 [ Netlify: React 19 Frontend ]
      │ (HTTPS REST API calls)
      ▼
 [ Render: FastAPI + PyTorch CPU + DeBERTa NLI Backend ]
      │
      ├── arXiv API
      ├── Semantic Scholar API
      ├── OpenAlex API
      └── Cloud LLM (Gemini 2.0 Flash / Qwen)
```

---

## Step 1: Deploy Backend to Render

### Option A: 1-Click Blueprint (Recommended)
1. Push your repository to GitHub / GitLab.
2. Log into [Render Dashboard](https://dashboard.render.com).
3. Click **New +** $\rightarrow$ **Blueprint**.
4. Connect your repository. Render will automatically detect [`render.yaml`](./render.yaml).
30: 5. Fill in the required environment variables:
31:    - `OPENAI_API_KEY`: Your OpenAI API key (sk-...)
32:    - `GEMINI_API_KEY` (Optional): Google Gemini API key
33:    - `SEMANTIC_SCHOLAR_API_KEY` (Optional): Semantic Scholar API key
34:    - `OPENALEX_API_KEY` (Optional): OpenAlex API key
35: 6. Click **Apply**. Render will build and deploy the service.
36: 
37: ### Option B: Manual Web Service Setup
38: If creating manually:
39: 1. In Render Dashboard, click **New +** $\rightarrow$ **Web Service**.
40: 2. Connect your Git repository.
41: 3. Configure the following settings:
42:    - **Name**: `researchlens-backend`
43:    - **Language**: `Python 3`
44:    - **Region**: Any (e.g. `Oregon (US West)` or `Frankfurt (EU)`)
45:    - **Branch**: `main` (or your active branch)
46:    - **Root Directory**: *(Leave empty / root)*
47:    - **Build Command**:
48:      ```bash
49:      pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r backend/requirements.txt
50:      ```
51:      *(Note: Using the PyTorch CPU index avoids downloading the ~2.5GB CUDA packages, keeping the build fast and within standard disk limits)*
52:    - **Start Command**:
53:      ```bash
54:      uvicorn backend.main:app --host 0.0.0.0 --port $PORT
55:      ```
56:    - **Plan**: `Starter` (or `Standard` recommended for NLI DeBERTa model memory; `Free` has a 512MB RAM cap).
57: 
58: 4. Add **Environment Variables** in the Render settings:
59:    | Key | Value | Description |
60:    | :--- | :--- | :--- |
61:    | `PYTHON_VERSION` | `3.11.9` | Python runtime version |
62:    | `HOST` | `0.0.0.0` | Bind address |
63:    | `LLM_PROVIDER` | `openai` | Primary cloud LLM provider (`openai`, `gemini`, `qwen`) |
64:    | `OPENAI_API_KEY` | `sk-...` | Your OpenAI API Key |
65:    | `OPENAI_MODEL` | `gpt-4o-mini` | Fast, cost-effective OpenAI model |
66:    | `NLI_DEVICE` | `cpu` | Uses CPU for DeBERTa inference |
   | `CORS_ORIGINS` | `*` | Or specify your Netlify domain |
   | `OPENALEX_EMAIL` | `your-email@example.com` | Polite pool identification |
   | `SEMANTIC_SCHOLAR_API_KEY` | *(Optional)* | Higher rate limits for S2 |

5. Once deployed, copy your backend service URL (e.g. `https://researchlens-backend.onrender.com`).
6. Test it in your browser: `https://researchlens-backend.onrender.com/health` should return `{"status": "ok", ...}`.

---

## Step 2: Deploy Frontend to Netlify

1. Log into [Netlify Dashboard](https://app.netlify.com).
2. Click **Add new site** $\rightarrow$ **Import an existing project**.
3. Select **GitHub** and choose your repository.
4. Netlify will automatically detect [`netlify.toml`](./netlify.toml):
   - **Build command**: `npm run build`
   - **Publish directory**: `dist`
5. Go to **Environment variables** $\rightarrow$ **Add a variable**:
   | Key | Value |
   | :--- | :--- |
   | `VITE_API_URL` | `https://researchlens-backend.onrender.com` |
   *(Replace with your actual Render URL from Step 1 — do not include a trailing slash)*
6. Click **Deploy site**.
7. Once deployed, open your Netlify URL (e.g. `https://researchlens.netlify.app`).

---

## Step 3: Verify Integration

1. Open your Netlify frontend in the browser.
2. In the Search bar, run a test query (e.g., *"What are the most effective techniques for reducing the computational cost and memory usage of large language models during inference?"*).
3. Verify that:
   - The job starts immediately without CORS errors in DevTools Console.
   - Live pipeline progress steps illuminate.
   - Retrieved papers and verified claims populate.
   - Final report renders with the Research Confidence badge.

---

## Troubleshooting

### CORS Errors
- The backend automatically permits all `https://*.netlify.app` subdomains.
- If you use a custom domain on Netlify (e.g. `https://research.yourdomain.com`), add it to `CORS_ORIGINS` on Render:
  `CORS_ORIGINS=https://research.yourdomain.com`

### Free Tier Render Spin-Down ("Cold Starts")
- Render's free tier spins down inactive web services after 15 minutes.
- The initial request after inactivity may take ~30–50 seconds to boot up.
- Setting `healthCheckPath: /health` in Render or using an uptime monitor keeps the instance warm.
