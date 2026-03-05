"""
Embed.py
========
Utilities for creating L2-normalised 768-dim fashion image embeddings.

Architecture
------------
  Input Image  →  FashionCLIP ViT-B/32 Vision Encoder
               →  768-Dimensional Pooler Output  (raw Vision Transformer CLS token)
               →  L2 Normalisation
               →  FAISS Inner-Product Index  (inner-product on unit vectors = cosine similarity)

Why 768-dim pooler, not 512-dim CLIP projection?
- The ViT-B/32 vision backbone natively produces a 768-dim representation.
- CLIP's visual_projection layer compresses it to 512-dim for the joint image-text space.
- For *image-to-image* retrieval we don't need the joint text space — the raw 768-dim
  features retain more visual detail and match the standard ViT architecture.
- FashionCLIP weights are still used, so the encoder is fashion-domain-tuned.
"""

import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset
from huggingface_hub import create_repo

MODEL_CKPT = "patrickjohncyh/fashion-clip"

# ---------------------------------------------------------------------------
# Per-image helper (kept for backwards compatibility & single-image use)
# ---------------------------------------------------------------------------

def extract_embeddings(image, processor, model):
    """
    Extract an L2-normalised 768-dim fashion embedding from a single PIL image.

    Uses the raw ViT-B/32 pooler output (768-dim) — the Vision Transformer's
    CLS token representation before CLIP's projection to 512-dim.
    Returns a 1-D numpy array of shape (768,).
    """
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        vision_out = model.vision_model(pixel_values=inputs["pixel_values"])
        features = vision_out.pooler_output  # (1, 768) — raw ViT CLS token
    vec = features.squeeze().numpy()
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


# ---------------------------------------------------------------------------
# Batch helper (used by upload_ds.py and build_local_index.py)
# ---------------------------------------------------------------------------

def extract_embeddings_batch(batch, processor, model, device="cpu"):
    """
    Process a HuggingFace dataset batch dict.
    Expects batch["image"] to be a list of PIL Images.
    Returns a dict with key "embeddings" (list of 1-D float lists, L2-normed, 768-dim).

    Extracts the raw ViT-B/32 pooler output (768-dim CLS token) instead of
    the projected 512-dim CLIP embedding — more visual detail for image-to-image search.
    """
    images = [img.convert("RGB") for img in batch["image"]]
    inputs = processor(images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        vision_out = model.vision_model(pixel_values=inputs["pixel_values"])
        feats = vision_out.pooler_output  # (B, 768) — raw ViT CLS token

    feats = feats.cpu().numpy().astype("float32")
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    feats = feats / np.maximum(norms, 1e-8)          # L2 normalise

    return {"embeddings": feats.tolist()}


# ---------------------------------------------------------------------------
# Full pipeline: load dataset → embed → push to Hub
# ---------------------------------------------------------------------------

def create_dataset_embeddings(
    input_dataset: str,
    output_dataset: str,
    token: str,
    model_ckpt: str = MODEL_CKPT,
    batch_size: int = 32,
):
    """
    Load *input_dataset* from the HuggingFace Hub, compute L2-normalised
    FashionCLIP embeddings for every image, then push the result to
    *output_dataset*.

    Parameters
    ----------
    input_dataset  : HF repo id of the source image dataset
    output_dataset : HF repo id where the embedding dataset will be pushed
    token          : HuggingFace write token
    model_ckpt     : FashionCLIP checkpoint to use
    batch_size     : images per batch (reduce if you run out of RAM)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model '{model_ckpt}' on {device.upper()} …")

    processor = CLIPProcessor.from_pretrained(model_ckpt)
    model = CLIPModel.from_pretrained(model_ckpt).to(device)
    model.eval()

    print(f"Loading dataset '{input_dataset}' …")
    dataset = load_dataset(input_dataset, split="train")

    def _embed(batch):
        return extract_embeddings_batch(batch, processor, model, device)

    print(f"Computing embeddings (batch_size={batch_size}) …")
    dataset_with_embeddings = dataset.map(
        _embed,
        batched=True,
        batch_size=batch_size,
        desc="Embedding",
    )

    # Push to hub
    try:
        create_repo(output_dataset, token=token, repo_type="dataset", exist_ok=True)
    except Exception as e:
        if "409" in str(e):
            print(f"Repository '{output_dataset}' already exists, updating …")
        else:
            raise

    print(f"Pushing to '{output_dataset}' …")
    dataset_with_embeddings.push_to_hub(output_dataset, token=token)
    print("Done.")

