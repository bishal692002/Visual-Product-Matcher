"""
build_local_index.py
====================
Downloads `ashraq/fashion-product-images-small` (44 072 fashion images, freely
available on HuggingFace Hub) and builds a LOCAL 768-dim embedding index that
app.py will automatically pick up on the next run.

Architecture (matches system diagram)
--------------------------------------
  Input Image
      └─► FashionCLIP ViT-B/32 Vision Encoder  (fashion-domain fine-tuned)
              └─► 768-Dimensional Pooler Output  (raw CLS token, L2-normalised)
                      └─► FAISS Inner-Product Index  ≡  Cosine Similarity Search

Why 768-dim?
  The ViT-B/32 backbone produces a 768-dim CLS token.  CLIP's visual_projection
  compresses it to 512-dim for the joint image-text space.  For image-to-image
  retrieval we don't need that joint space — the raw 768-dim pooler retains
  more visual detail (color, texture, garment shape).

No HuggingFace write token is required — everything stays on your machine.

Usage:
    python build_local_index.py          # full 44K images  (recommended)
    python build_local_index.py --limit 500   # quick smoke test

Output:
    ./cache/fashion_index/   ← HF Dataset with 768-dim embeddings + metadata

Time estimate (Apple MPS / CUDA):  ~14 min for 44K images
Time estimate (CPU only):          ~35 min for 44K images
"""

import os
import sys
import argparse
import numpy as np
import torch
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SOURCE_DATASET  = "ashraq/fashion-product-images-small"  # 44K images, public
OUTPUT_DIR      = os.path.join(os.path.dirname(__file__), "cache", "fashion_index")
MODEL_CKPT      = "patrickjohncyh/fashion-clip"   # fashion-domain CLIP
BATCH_SIZE      = 32  # lower if you run out of RAM

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Build local FashionCLIP index from 44K dataset")
parser.add_argument("--limit",  type=int, default=None,
                    help="Process only first N images (useful for quick tests)")
parser.add_argument("--batch",  type=int, default=BATCH_SIZE,
                    help=f"Batch size for embedding (default {BATCH_SIZE})")
parser.add_argument("--source", type=str, default=SOURCE_DATASET)
parser.add_argument("--output", type=str, default=OUTPUT_DIR)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

# Apple Silicon MPS > CUDA > CPU
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🤖  Model    : {MODEL_CKPT}")
print(f"   Device   : {device.upper()}")
print(f"   Batch    : {args.batch}")

processor = CLIPProcessor.from_pretrained(MODEL_CKPT)
model     = CLIPModel.from_pretrained(MODEL_CKPT).to(device)
model.eval()

# ---------------------------------------------------------------------------
# Load source dataset (streaming-friendly via HF Hub, no auth needed)
# ---------------------------------------------------------------------------

print(f"\n📦  Source   : {args.source}")
dataset = load_dataset(args.source, split="train")

if args.limit:
    dataset = dataset.select(range(min(args.limit, len(dataset))))

print(f"   Images   : {len(dataset):,}")

if "image" not in dataset.column_names:
    sys.exit(f"ERROR: source dataset has no 'image' column. Found: {dataset.column_names}")

# ---------------------------------------------------------------------------
# Batch embedding with FashionCLIP
# ---------------------------------------------------------------------------

def embed_batch(batch):
    """
    Accepts a HuggingFace batch dict.
    Returns {'embeddings': list[list[float]]}  — L2-normalised 768-dim vectors.

    Extracts the raw ViT-B/32 pooler output (768-dim CLS token) instead of
    the projected 512-dim CLIP embedding, giving richer visual features for
    image-to-image search (colour, texture, garment shape).
    """
    images = [img.convert("RGB") for img in batch["image"]]
    inputs = processor(images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        vision_out = model.vision_model(pixel_values=inputs["pixel_values"].to(device))
        feats = vision_out.pooler_output   # (B, 768) — raw ViT CLS token

    feats = feats.cpu().numpy().astype("float32")
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    feats = feats / np.maximum(norms, 1e-8)          # L2 normalise

    return {"embeddings": feats.tolist()}

print("\n🧠  Computing FashionCLIP embeddings …")
dataset_with_emb = dataset.map(
    embed_batch,
    batched=True,
    batch_size=args.batch,
    desc="Embedding",
)

print(f"\n   Columns  : {dataset_with_emb.column_names}")

# ---------------------------------------------------------------------------
# Save to disk (HuggingFace Dataset format — app.py loads it with load_dataset)
# ---------------------------------------------------------------------------

os.makedirs(args.output, exist_ok=True)
print(f"\n💾  Saving to : {args.output}")
dataset_with_emb.save_to_disk(args.output)

print(f"\n✅  Done!  {len(dataset_with_emb):,} fashion images indexed with FashionCLIP.")
print(f"   Index saved to: {args.output}")
print()
print("Restart app.py / main.py — they will automatically use the new 44K index.")
