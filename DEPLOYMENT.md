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
5. Fill in the required environment variables:
   - `GEMINI_API_KEY`: Your Google Gemini API key (recommended for cloud inference)
   - `SEMANTIC_SCHOLAR_API_KEY` (Optional): Semantic Scholar API key
   - `OPENALEX_API_KEY` (Optional): OpenAlex API key
6. Click **Apply**. Render will build and deploy the service.

### Option B: Manual Web Service Setup
If creating manually:
1. In Render Dashboard, click **New +** $\rightarrow$ **Web Service**.
2. Connect your Git repository.
3. Configure the following settings:
   - **Name**: `researchlens-backend`
   - **Language**: `Python 3`
   - **Region**: Any (e.g. `Oregon (US West)` or `Frankfurt (EU)`)
   - **Branch**: `main` (or your active branch)
   - **Root Directory**: *(Leave empty / root)*
   - **Build Command**:
     ```bash
     pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r backend/requirements.txt
     ```
     *(Note: Using the PyTorch CPU index avoids downloading the ~2.5GB CUDA packages, keeping the build fast and within standard disk limits)*
   - **Start Command**:
     ```bash
     uvicorn backend.main:app --host 0.0.0.0 --port $PORT
     ```
   - **Plan**: `Starter` (or `Standard` recommended for NLI DeBERTa model memory; `Free` has a 512MB RAM cap).

4. Add **Environment Variables** in the Render settings:
   | Key | Value | Description |
   | :--- | :--- | :--- |
   | `PYTHON_VERSION` | `3.11.9` | Python runtime version |
   | `HOST` | `0.0.0.0` | Bind address |
   | `LLM_PROVIDER` | `gemini` | Cloud LLM provider (`gemini` or `qwen`) |
   | `GEMINI_API_KEY` | `AIzaSy...` | Your Gemini API Key from Google AI Studio |
   | `GEMINI_MODEL` | `gemini-2.0-flash` | Lightweight, fast cloud model |
   | `NLI_DEVICE` | `cpu` | Uses CPU for DeBERTa inference |
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
