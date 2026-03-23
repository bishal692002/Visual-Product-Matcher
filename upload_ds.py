"""
upload_ds.py
============
Builds a 20K+ fashion-product embeddings dataset on HuggingFace Hub.

Source  : ashraq/fashion-product-images-small  (~44 K images, same metadata
          schema as the original Kaggle Fashion Product Images dataset)
Embeds  : patrickjohncyh/fashion-clip (768-dim pooler output)
Output  : <HF_USERNAME>/fashion-products-embeddings   (embeddings + metadata)

Usage:
    export HUGGINGFACE_TOKEN=hf_xxx
    python upload_ds.py

    # To limit to the first N images during testing:
    python upload_ds.py --limit 500
"""

import os
import sys
import argparse
import numpy as np
import torch
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel
from huggingface_hub import create_repo

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Source dataset (44 K fashion product images, freely available on HF Hub)
SOURCE_DATASET = "ashraq/fashion-product-images-small"

# Your HuggingFace output repository  (set HF_USER env var or edit directly)
HF_USER = os.getenv("HF_USER", "Gauravannad")
OUTPUT_REPO = f"{HF_USER}/fashion-products-embeddings"

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "your-token-here")

# Embedding model (same model used in app.py / main.py)
MODEL_CKPT = "patrickjohncyh/fashion-clip"   # fashion-domain CLIP \u2014 understands color & style

# Batch size for embedding (reduce if you run out of RAM / VRAM)
BATCH_SIZE = 64

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=None,
                    help="Process only first N images (useful for quick tests)")
parser.add_argument("--source", type=str, default=SOURCE_DATASET,
                    help="HuggingFace source dataset repo id")
parser.add_argument("--output", type=str, default=OUTPUT_REPO,
                    help="HuggingFace output repo id")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

print(f"🤖  Loading model  : {MODEL_CKPT}")
processor = CLIPProcessor.from_pretrained(MODEL_CKPT)
model = CLIPModel.from_pretrained(MODEL_CKPT)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"   Running on      : {device.upper()}")

# ---------------------------------------------------------------------------
# Load source dataset
# ---------------------------------------------------------------------------

print(f"\n📦  Loading source dataset : {args.source}")
dataset = load_dataset(args.source, split="train")

if args.limit:
    dataset = dataset.select(range(min(args.limit, len(dataset))))

print(f"   Total images    : {len(dataset):,}")

# Make sure the dataset has an 'image' column (PIL Image)
if "image" not in dataset.column_names:
    raise ValueError(
        f"Source dataset must have an 'image' column. "
        f"Found: {dataset.column_names}"
    )

# ---------------------------------------------------------------------------
# Batch embedding function
# ---------------------------------------------------------------------------

def embed_batch(batch):
    """Embed a batch of PIL images with FashionCLIP; returns L2-normalised 768-dim vectors."""
    images = [img.convert("RGB") for img in batch["image"]]
    inputs = processor(images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        vision_out = model.vision_model(pixel_values=inputs["pixel_values"])
        cls_vectors = vision_out.pooler_output  # (B, 768)

    cls_vectors = cls_vectors.cpu().numpy().astype("float32")   # (B, 768)

    # L2 normalise so inner-product == cosine similarity in FAISS
    norms = np.linalg.norm(cls_vectors, axis=1, keepdims=True)
    cls_vectors = cls_vectors / np.maximum(norms, 1e-8)

    return {"embeddings": cls_vectors.tolist()}

# ---------------------------------------------------------------------------
# Compute embeddings
# ---------------------------------------------------------------------------

print(f"\n🧠  Computing embeddings  (batch_size={BATCH_SIZE}) …")
dataset_with_emb = dataset.map(
    embed_batch,
    batched=True,
    batch_size=BATCH_SIZE,
    desc="Embedding images",
)
print(f"   Done. Columns   : {dataset_with_emb.column_names}")

# ---------------------------------------------------------------------------
# Push to Hub
# ---------------------------------------------------------------------------

print(f"\n📤  Pushing to HuggingFace Hub : {args.output}")
try:
    create_repo(args.output, token=HF_TOKEN, repo_type="dataset", exist_ok=True)
except Exception as e:
    print(f"   Repo note: {e}")

dataset_with_emb.push_to_hub(args.output, token=HF_TOKEN)

print(f"\n✅  Done!  {len(dataset_with_emb):,} images with embeddings uploaded to:")
print(f"   https://huggingface.co/datasets/{args.output}")
print()
print("Next step — update DATASET_REPO in app.py and main.py to:")
print(f'   DATASET_REPO = "{args.output}"')




