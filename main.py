from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import base64
import io
import os
import pandas as pd
import torch

# Import necessary libraries from Hugging Face
from datasets import load_dataset, load_from_disk
from transformers import CLIPProcessor, CLIPModel

# --- 1. Application Setup ---
# Initialize FastAPI app
app = FastAPI(title="Visual Product Recommendation API")

# Add CORS middleware to allow frontend-backend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load product metadata
print("Loading product metadata...")
try:
    metadata_df = pd.read_csv('./img/styles.csv', on_bad_lines='skip')
    print(f"Metadata loaded: {len(metadata_df)} products")
    METADATA_LOADED = True
except Exception as e:
    print(f"ERROR: Failed to load metadata: {e}")
    metadata_df = None
    METADATA_LOADED = False

# --- 2. Model and Dataset Loading (Global) ---
# This section runs only once when the application starts.
# Loading models and data is time-consuming, so we do it here to avoid
# reloading on every API request, which would be very slow.

print("Loading FashionCLIP model...")
# FashionCLIP is fine-tuned on ~700K fashion image-text pairs.
# It understands garment COLOR, type, and style — far more accurate than ViT-Base for apparel search.
MODEL_CKPT = "patrickjohncyh/fashion-clip"
processor = CLIPProcessor.from_pretrained(MODEL_CKPT)
model = CLIPModel.from_pretrained(MODEL_CKPT)
model.eval()
# Apple Silicon MPS > CUDA > CPU
_device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(_device)
print(f"FashionCLIP model loaded on {_device.upper()}.")

print("Loading embeddings dataset...")
# Try local 44K index first (built by build_local_index.py), then fall back to HF hub
LOCAL_INDEX_PATH = os.path.join(os.path.dirname(__file__), "cache", "fashion_index")
DATASET_REPO = "bishal692002/fashion-products-embeddings"  # fallback (small)
try:
    if os.path.exists(LOCAL_INDEX_PATH):
        dataset = load_from_disk(LOCAL_INDEX_PATH)
        print(f"Loaded local index: {len(dataset):,} images")
    else:
        dataset = load_dataset(DATASET_REPO, split="train")
        print(f"Loaded HF fallback dataset: {len(dataset):,} images")
        print("WARNING: Run build_local_index.py for the full 44K fashion dataset!")

    # Normalise embeddings to unit vectors (L2) so inner-product == cosine sim
    raw_embs = np.array(dataset["embeddings"]).astype("float32")
    norms = np.linalg.norm(raw_embs, axis=1, keepdims=True)
    normed = raw_embs / np.maximum(norms, 1e-8)
    if "embeddings_norm" not in dataset.column_names:
        dataset = dataset.add_column("embeddings_norm", normed.tolist())

    # Add FAISS inner-product index (cosine similarity for unit vectors)
    print("Adding FAISS cosine-similarity index...")
    dataset.add_faiss_index(column="embeddings_norm", metric_type=0)  # 0 = METRIC_INNER_PRODUCT
    print("FAISS index added successfully.")
    DATASET_LOADED = True
except Exception as e:
    print(f"ERROR: Failed to load dataset or add FAISS index: {e}")
    DATASET_LOADED = False

# --- 3. Helper Functions ---

def extract_embeddings(image: Image.Image) -> np.ndarray:
    """
    Extract an L2-normalised FashionCLIP image embedding (512-dim).
    FashionCLIP understands garment color, type and style accurately.
    """
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(_device)
    with torch.no_grad():
        vision_out = model.vision_model(pixel_values=pixel_values)
        features = model.visual_projection(vision_out.pooler_output)  # (1, 512)
    vec = features.squeeze().cpu().numpy()  # always back to CPU for FAISS
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

def pil_to_base64(image: Image.Image) -> str:
    """Converts a PIL Image object to a Base64 encoded string."""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


# --- 4. API Endpoint ---

@app.post("/recommend/")
async def recommend_products(file: UploadFile = File(...)):
    """
    Receives an uploaded image, finds similar products, and returns them with metadata.
    """
    if not DATASET_LOADED:
        return JSONResponse(
            status_code=503, 
            content={"error": "Server is not ready. Dataset not loaded. Please try again later."}
        )

    try:
        # Read the uploaded image file
        contents = await file.read()
        
        # Validate file size (max 10MB)
        if len(contents) > 10 * 1024 * 1024:
            return JSONResponse(
                status_code=400,
                content={"error": "File too large. Maximum size is 10MB."}
            )
        
        # Try to open image
        try:
            uploaded_image = Image.open(io.BytesIO(contents))
            # Convert to RGB if needed
            if uploaded_image.mode != 'RGB':
                uploaded_image = uploaded_image.convert('RGB')
        except Exception as img_error:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid image file. Please upload a valid JPG, PNG, or JPEG image. Details: {str(img_error)}"}
            )

        # Generate an embedding for the uploaded image
        try:
            query_embedding = extract_embeddings(uploaded_image)
        except Exception as emb_error:
            return JSONResponse(
                status_code=500,
                content={"error": f"Failed to process image. Please try a different image. Details: {str(emb_error)}"}
            )

        # Use the FAISS index to find the 10 most similar images
        try:
            scores, retrieved_examples = dataset.get_nearest_examples(
                "embeddings_norm", query_embedding, k=10
            )
            print(f"Debug: cosine similarity scores: {scores[:3]}")  # 1.0 = identical
        except Exception as search_error:
            return JSONResponse(
                status_code=500,
                content={"error": f"Failed to search for similar products. Please try again. Details: {str(search_error)}"}
            )

        # Prepare the results to be sent back as JSON
        recommendations = []
        for i in range(len(retrieved_examples["image"])):
            try:
                img_base64 = pil_to_base64(retrieved_examples["image"][i])
                similarity_score = round(max(0.0, min(1.0, float(scores[i]))) * 100, 2)

                # Pull metadata directly from the dataset columns (ashraq dataset has all fields)
                def _col(col):
                    data = retrieved_examples.get(col)
                    if data is None or i >= len(data):
                        return "Unknown"
                    val = data[i]
                    return str(val) if val is not None and str(val) not in ("", "nan", "None") else "Unknown"

                product_info = {
                    "image":            img_base64,
                    "similarity_score": similarity_score,
                    "product_name":     _col("productDisplayName"),
                    "category":         _col("masterCategory"),
                    "sub_category":     _col("subCategory"),
                    "article_type":     _col("articleType"),
                    "color":            _col("baseColour"),
                    "gender":           _col("gender"),
                    "season":           _col("season"),
                    "usage":            _col("usage"),
                }
                recommendations.append(product_info)
            except Exception as rec_error:
                print(f"Error processing recommendation {i}: {rec_error}")
                continue

        # Return the list of recommendations with metadata
        return JSONResponse(content={"recommendations": recommendations, "total": len(recommendations)})
    
    except Exception as e:
        print(f"Unexpected error in recommend_products: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"An unexpected error occurred. Please try again. Details: {str(e)}"}
        )

@app.get("/")
def read_root():
    """Root endpoint with API status information"""
    return {
        "status": "API is running",
        "message": "POST to /recommend/ to get recommendations",
        "dataset_loaded": DATASET_LOADED,
        "metadata_loaded": METADATA_LOADED,
        "endpoints": {
            "health": "/health",
            "recommend": "/recommend/",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy" if DATASET_LOADED else "degraded",
        "dataset": "loaded" if DATASET_LOADED else "not loaded",
        "metadata": "loaded" if METADATA_LOADED else "not loaded",
        "model": MODEL_CKPT,
        "version": "1.0.0"
    }
