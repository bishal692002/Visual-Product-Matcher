# Visual Product Matcher

AI-powered fashion similarity search with two runnable interfaces:
- FastAPI + HTML/CSS/JS UI (templates/static) for Render deployment
- Streamlit UI (`app.py`) for Streamlit deployment

The system is designed to work with a 44k-image embeddings dataset.

## What Was Cleaned For Deployment

This repository was cleaned to keep runtime files only:
- Removed non-runtime test files and test artifacts.
- Added stricter ignore rules for local caches and tooling artifacts.
- Added `render.yaml` for one-click Render setup.

## Project Structure

- `main.py`: FastAPI backend and HTML/CSS/JS UI server
- `templates/`, `static/`: Production UI assets
- `app.py`: Streamlit app
- `build_local_index.py`: Build 44k local embeddings index
- `upload_ds.py`: Push embeddings dataset to Hugging Face Hub
- `services/`, `detection/`: Outfit analysis and recommendation logic
- `render.yaml`: Render deployment config

## Local Run

### 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. FastAPI + HTML UI

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open: `http://localhost:8000`

### 3. Streamlit UI

```bash
streamlit run app.py
```

Open: `http://localhost:8501`

## 44k Dataset Strategy (Recommended)

Do **not** commit local `cache/` to GitHub.

Use one of these options:

### Option A (best for deployment): host embeddings on Hugging Face Dataset

1. Build local 44k embeddings once:

```bash
python build_local_index.py
```

2. Push to your HF dataset repo:

```bash
python -c "from datasets import load_from_disk; ds=load_from_disk('cache/fashion_index'); ds.push_to_hub('YOUR_HF_USERNAME/fashion-products-embeddings-44k')"
```

3. Set env var in deployment:

- `EMBEDDINGS_DATASET_REPO=YOUR_HF_USERNAME/fashion-products-embeddings-44k`

The app auto-loads this repo when local cache is absent.

### Option B: local cache only

Keep `cache/fashion_index` only on your machine/server. Not suitable for ephemeral free hosts.

## Deploy To Render (Free) - HTML/CSS/JS UI

This deploys your **FastAPI + templates/static UI**.

### Steps

1. Push this repo to GitHub.
2. In Render: New -> Web Service -> Connect repo.
3. Render reads `render.yaml` automatically.
4. Set environment variable in Render dashboard:
   - `EMBEDDINGS_DATASET_REPO=YOUR_HF_USERNAME/fashion-products-embeddings-44k`
5. Deploy.

### Runtime details

- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Free tier may sleep when idle.
- Cold start can be slow due to model/index loading.

## Deploy To Streamlit Community Cloud (Free)

This deploys the **Streamlit UI** (`app.py`).

### Steps

1. Push repo to GitHub.
2. In Streamlit Cloud: New app.
3. Set:
   - Main file path: `app.py`
4. In app settings -> Secrets / environment:
   - `EMBEDDINGS_DATASET_REPO="YOUR_HF_USERNAME/fashion-products-embeddings-44k"`
5. Deploy.

## Important Free-Tier Notes

- Free deployments are usually ephemeral and can sleep.
- Avoid rebuilding 44k index on each boot.
- Always host embeddings in a remote dataset repo for reliable startup.

## Recommended Production-Like Setup (Still Free)

- GitHub: source code
- Hugging Face Dataset: 44k embeddings repo
- Render (or Streamlit Cloud): app runtime
- Env var: `EMBEDDINGS_DATASET_REPO`

## Quick Deployment Checklist

1. `cache/` is ignored in `.gitignore`.
2. 44k embeddings pushed to HF dataset repo.
3. `EMBEDDINGS_DATASET_REPO` configured in host.
4. App starts and `/health` is healthy.
5. URL mode and upload mode both tested in UI.

## License

MIT (see `LICENSE`).
