# Visual Product Matcher

Visual Product Matcher is an AI-powered fashion retrieval project that finds visually similar products from outfit or single-item images.

It supports two interfaces in the same codebase:
- FastAPI + HTML/CSS/JS frontend
- Streamlit app

The retrieval pipeline uses FashionCLIP embeddings with FAISS nearest-neighbor search over a 44k-item index.

## Features

- Single item similarity search from upload or image URL
- Full outfit analysis (top, bottom, shoes)
- Per-category search results for detected outfit regions
- Outfit completion recommendations
- FastAPI endpoints for integration with custom frontends
- Local 44k index support for faster runtime

## Tech Stack

- Python, FastAPI, Streamlit
- FashionCLIP (`patrickjohncyh/fashion-clip`)
- FAISS (cosine similarity over normalized vectors)
- YOLOv8 (`ultralytics`) for outfit-region detection
- Hugging Face Datasets for index storage/loading

## Project Structure

- `main.py`: FastAPI application and API routes
- `templates/`, `static/`: HTML/CSS/JS frontend
- `app.py`: Streamlit app interface
- `services/outfit_search.py`: outfit analysis orchestration
- `services/recommendation.py`: outfit recommendation logic
- `detection/outfit_detector.py`: detection + category zoning
- `build_local_index.py`: builds local 44k embeddings index
- `upload_ds.py`: uploads embeddings dataset to Hugging Face
- `img/styles.csv`: metadata used for enrichment/recommendations

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

### FastAPI + HTML UI

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`

### Streamlit

```bash
streamlit run app.py
```

Open `http://localhost:8501`

## Data and Index

The system can run with:
- Local index at `cache/fashion_index`
- Remote embeddings dataset from Hugging Face

If local index does not exist, the app falls back to Hugging Face dataset loading.

Environment variable:

```bash
EMBEDDINGS_DATASET_REPO=<your_hf_dataset_repo>
```

### Build local 44k index

```bash
python build_local_index.py
```

Quick test build:

```bash
python build_local_index.py --limit 500
```

## API Overview

- `GET /`: frontend home
- `GET /health`: service health
- `GET /api-status`: component status
- `POST /recommend/`: similarity search from uploaded file
- `POST /recommend-url/`: similarity search from image URL
- `POST /outfit-analysis/`: outfit analysis from uploaded file
- `POST /outfit-analysis-url/`: outfit analysis from image URL

## Notes

- The local `cache/` index is intentionally ignored by git.
- First startup may be slower due to model/index loading.
- URL-based analysis is handled server-side for better reliability.

## License

MIT. See `LICENSE`.
