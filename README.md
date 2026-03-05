# Visual Product Matcher

> AI-powered fashion similarity search — upload any product image and instantly find visually similar items from a 44K-image catalog.

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

Visual Product Matcher is a full-stack AI application that lets users find visually similar fashion products. Upload a photo or paste an image URL, and the system returns the closest matches ranked by cosine similarity — in under a second.

The system combines **FashionCLIP** (a domain-tuned vision-language model), **FAISS** for sub-millisecond approximate nearest-neighbor search, and a **Streamlit** frontend — all connected through a **FastAPI** backend.

---

## Features

- **Image upload or URL input** — flexible input methods for end users
- **FashionCLIP embeddings** — a ViT-B/32 model fine-tuned on 700K+ fashion image-text pairs, giving it deep understanding of garment color, type, and style
- **44K+ product catalog** — sourced from `ashraq/fashion-product-images-small` on HuggingFace Hub
- **FAISS cosine-similarity search** — blazing-fast approximate nearest-neighbor retrieval
- **Rich metadata display** — product name, category, sub-category, color, gender, season, and usage for every result
- **Similarity threshold filter** — adjustable 0–100% confidence slider
- **Apple MPS / CUDA / CPU support** — automatically selects the best available device
- **REST API backend** — independent FastAPI service at `/recommend/` for integration with other frontends

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit + custom CSS |
| Backend API | FastAPI + Uvicorn |
| Embedding model | `patrickjohncyh/fashion-clip` (FashionCLIP) |
| Vector search | FAISS (inner-product / cosine similarity) |
| Dataset | HuggingFace Hub — `ashraq/fashion-product-images-small` |
| Image processing | Pillow, Transformers |
| Acceleration | PyTorch (MPS / CUDA / CPU) |

---

## Project Structure

```
Visual-Product-Matcher/
├── app.py                  # Streamlit frontend (standalone — embeds model inline)
├── main.py                 # FastAPI backend  (serves /recommend/ endpoint)
├── build_local_index.py    # One-time script: downloads 44K dataset & builds FAISS index
├── Embed.py                # Shared embedding utilities (used by build_local_index & upload_ds)
├── upload_ds.py            # Utility: embed dataset and push to HuggingFace Hub
├── evaluate_search.py      # Offline benchmark: FAISS vs brute-force retrieval metrics
├── requirements.txt        # Python dependencies
├── .gitignore
├── LICENSE
├── CONTRIBUTING.md
└── img/
    └── styles.csv          # Product metadata (id, name, category, color, gender …)
```

---

## Installation

### Prerequisites

- Python 3.9 or higher
- 6 GB RAM minimum (8 GB recommended for the full 44K index)
- Internet connection (first run downloads the model and dataset)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/bishal692002/Visual-Product-Matcher.git
cd Visual-Product-Matcher

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Option A — Streamlit app (recommended)

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

1. Choose **Upload Image** or **Image URL**
2. Select or paste a fashion product image
3. Adjust the **Minimum Similarity** slider and **Results to Show**
4. Click **Find Similar Products**
5. Browse results with similarity scores and product metadata

### Option B — FastAPI backend + any frontend

```bash
uvicorn main:app --reload
```

API is available at `http://localhost:8000`.

```bash
# Example: search with an image file
curl -X POST "http://localhost:8000/recommend/" \
     -H "accept: application/json" \
     -F "file=@your_image.jpg"
```

Interactive docs: `http://localhost:8000/docs`

### Build the local 44K index (optional but recommended)

The local index gives faster startup and removes the HuggingFace Hub dependency at runtime.

```bash
python build_local_index.py            # full 44K images (~14 min on Apple MPS / CUDA)
python build_local_index.py --limit 500  # quick smoke test
```

The index is saved to `cache/fashion_index/` and picked up automatically on the next run.

---

## Example Output

| Input | Top Match | Similarity |
|---|---|---|
| Blue denim jacket | Similar denim jacket (navy) | 94% |
| White sneakers | White low-top sports shoes | 91% |
| Red floral dress | Red printed midi dress | 89% |

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | API status and available endpoints |
| `/health` | GET | Health check — model and dataset status |
| `/recommend/` | POST | Upload image → JSON list of similar products |
| `/docs` | GET | Swagger UI (interactive API docs) |

### Response format (`/recommend/`)

```json
{
  "total": 10,
  "recommendations": [
    {
      "product_name": "Roadster Men Blue Jacket",
      "category": "Apparel",
      "sub_category": "Topwear",
      "article_type": "Jackets",
      "color": "Blue",
      "gender": "Men",
      "season": "Winter",
      "usage": "Casual",
      "similarity_score": 94.3,
      "image": "<base64-encoded JPEG>"
    }
  ]
}
```

---

## Future Improvements

- [ ] Text-to-image search (query by description, e.g. "red floral summer dress")
- [ ] Multi-image query support (combine embeddings from multiple reference images)
- [ ] User feedback loop to fine-tune ranking
- [ ] Streamlit Cloud / Docker deployment guide
- [ ] Outfit recommendation — find complementary items, not just similar ones
- [ ] Filter by category, gender, or price range

---

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

---

## Author

**Bishal Biswas** — [@bishal692002](https://github.com/bishal692002)

---

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [patrickjohncyh/fashion-clip](https://huggingface.co/patrickjohncyh/fashion-clip) — FashionCLIP model
- [ashraq/fashion-product-images-small](https://huggingface.co/datasets/ashraq/fashion-product-images-small) — source dataset
- [HuggingFace](https://huggingface.co) — model and dataset hosting
- [Streamlit](https://streamlit.io) — frontend framework
- [FAISS](https://github.com/facebookresearch/faiss) — vector search library
