from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import base64
import io
import pandas as pd

# Import necessary libraries from Hugging Face
from datasets import load_dataset
from transformers import AutoFeatureExtractor, AutoModel

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

print("Loading model and feature extractor...")
# Load the pre-trained Vision Transformer (ViT) model and its feature extractor
# This model is excellent for creating general-purpose image embeddings.
MODEL_CKPT = 'google/vit-base-patch16-224'
extractor = AutoFeatureExtractor.from_pretrained(MODEL_CKPT)
model = AutoModel.from_pretrained(MODEL_CKPT)
print("Model and feature extractor loaded successfully.")

print("Loading embeddings dataset from Hugging Face Hub...")
# IMPORTANT: Replace this with your own dataset repository from Step 2
# DATASET_REPO = "Gauravannad/fashion-dataset-embeddings"
DATASET_REPO = "Gauravannad/fashion-products-embeddings"
# Load the dataset containing pre-computed embeddings
try:
    dataset = load_dataset(DATASET_REPO, split="train")
    print("Dataset loaded successfully.")

    # Add a FAISS index to the 'embeddings' column for fast similarity search
    # This creates a highly optimized index for finding nearest neighbors.
    print("Adding FAISS index to the dataset...")
    dataset.add_faiss_index(column="embeddings")
    print("FAISS index added successfully.")
    DATASET_LOADED = True
except Exception as e:
    print(f"ERROR: Failed to load dataset or add FAISS index: {e}")
    DATASET_LOADED = False

# --- 3. Helper Functions ---

def extract_embeddings(image: Image.Image) -> np.ndarray:
    """Converts an image into a numerical vector (embedding)."""
    # Use the feature extractor to preprocess the image
    image_pp = extractor(image.convert("RGB"), return_tensors="pt")
    # Pass the preprocessed image to the model to get the features
    features = model(**image_pp).last_hidden_state[:, 0].detach().numpy()
    return features.squeeze()

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
                "embeddings", query_embedding, k=10
            )
            print(f"Debug: FAISS distances: {scores[:3]}")  # Show first 3 distances for debugging
        except Exception as search_error:
            return JSONResponse(
                status_code=500,
                content={"error": f"Failed to search for similar products. Please try again. Details: {str(search_error)}"}
            )

        # Prepare the results to be sent back as JSON
        recommendations = []
        for i in range(len(retrieved_examples["image"])):
            try:
                # Convert each recommended PIL image to a Base64 string
                img_base64 = pil_to_base64(retrieved_examples["image"][i])
                
                # Get product metadata if available
                file_name = retrieved_examples.get("file_name", [None])[i] if "file_name" in retrieved_examples else None
                product_id = retrieved_examples.get("id", [None])[i] if "id" in retrieved_examples else None
                
                # Calculate similarity score (lower distance = higher similarity)
                # FAISS returns L2 distances (typically 0-50 range for ViT embeddings)
                # Convert to similarity percentage: very similar images (distance ~5-10) should show 85-95%
                import math
                # Using exponential decay with scale=20 gives good range:
                # distance=0 -> 100%, distance=5 -> 95%, distance=10 -> 90%, distance=20 -> 78%
                similarity_score = math.exp(-scores[i] / 20)
                
                product_info = {
                    "image": img_base64,
                    "similarity_score": round(similarity_score * 100, 2),  # Percentage
                    "file_name": file_name
                }
                
                # Add metadata from CSV if available
                if METADATA_LOADED and metadata_df is not None:
                    try:
                        # Try to match by product_id first
                        product_row = None
                        if product_id is not None:
                            product_row = metadata_df[metadata_df['id'] == product_id]
                        
                        # If no match by ID, try to extract ID from filename
                        if (product_row is None or product_row.empty) and file_name:
                            # Extract numeric ID from filename (e.g., "12345.jpg" -> 12345)
                            import re
                            match = re.search(r'(\d+)', str(file_name))
                            if match:
                                file_id = int(match.group(1))
                                product_row = metadata_df[metadata_df['id'] == file_id]
                        
                        if product_row is not None and not product_row.empty:
                            product_info.update({
                                "product_name": str(product_row.iloc[0].get('productDisplayName', 'Unknown')),
                                "category": str(product_row.iloc[0].get('masterCategory', 'Unknown')),
                                "sub_category": str(product_row.iloc[0].get('subCategory', 'Unknown')),
                                "article_type": str(product_row.iloc[0].get('articleType', 'Unknown')),
                                "color": str(product_row.iloc[0].get('baseColour', 'Unknown')),
                                "gender": str(product_row.iloc[0].get('gender', 'Unknown')),
                            })
                    except Exception as meta_error:
                        print(f"Error retrieving metadata: {meta_error}")
                
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
