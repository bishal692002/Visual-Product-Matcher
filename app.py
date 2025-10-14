import streamlit as st
from PIL import Image
import requests
import io
import numpy as np
from urllib.parse import urlparse
import math
import pandas as pd

# Import ML libraries
from datasets import load_dataset
from transformers import AutoFeatureExtractor, AutoModel

# --- Page Configuration ---
st.set_page_config(
    page_title="Visual Product Matcher",
    page_icon="🛍️",
    layout="wide"
)

# --- Initialize Session State ---
@st.cache_resource
def load_model_and_dataset():
    """Load model and dataset once and cache them"""
    with st.spinner("🔄 Loading AI model and product database..."):
        # Load Vision Transformer model
        MODEL_CKPT = 'google/vit-base-patch16-224'
        extractor = AutoFeatureExtractor.from_pretrained(MODEL_CKPT)
        model = AutoModel.from_pretrained(MODEL_CKPT)
        
        # Load embeddings dataset from HuggingFace
        DATASET_REPO = "Gauravannad/fashion-products-embeddings"
        dataset = load_dataset(DATASET_REPO, split="train")
        
        # Add FAISS index for fast similarity search
        dataset.add_faiss_index(column="embeddings")
        
        # Load metadata
        try:
            metadata_df = pd.read_csv('./img/styles.csv', on_bad_lines='skip')
        except:
            metadata_df = None
        
        return extractor, model, dataset, metadata_df

# Load everything
extractor, model, dataset, metadata_df = load_model_and_dataset()

# --- Helper Functions ---
def extract_embeddings(image: Image.Image) -> np.ndarray:
    """Convert image to embedding vector"""
    image_pp = extractor(image.convert("RGB"), return_tensors="pt")
    features = model(**image_pp).last_hidden_state[:, 0].detach().numpy()
    return features.squeeze()

def is_valid_url(url):
    """Check if string is valid URL"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def load_image_from_url(url):
    """Download image from URL"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    except Exception as e:
        st.error(f"Failed to load image from URL: {e}")
        return None

def find_similar_products(query_image, top_k=10):
    """Find similar products using FAISS"""
    # Extract embedding from query image
    query_embedding = extract_embeddings(query_image)
    
    # Search in FAISS index
    scores, retrieved_examples = dataset.get_nearest_examples(
        "embeddings", query_embedding, k=top_k
    )
    
    # Prepare results
    results = []
    
    # Normalize scores to 0-100% range using min-max scaling
    # The closest item gets highest score, furthest gets lowest
    max_score = max(scores) if len(scores) > 0 else 1
    min_score = min(scores) if len(scores) > 0 else 0
    score_range = max_score - min_score if max_score != min_score else 1
    
    for i in range(len(retrieved_examples["image"])):
        # Calculate similarity: invert and normalize so closest = 100%, furthest = lower
        # Formula: 100 - ((distance - min) / (max - min) * 100)
        normalized_distance = (scores[i] - min_score) / score_range
        similarity_score = (1 - normalized_distance) * 100
        
        # Ensure it's in 0-100 range
        similarity_score = max(0, min(100, similarity_score))
        
        # Get metadata
        file_name = retrieved_examples.get("file_name", [None])[i]
        product_id = retrieved_examples.get("id", [None])[i]
        
        product_info = {
            "image": retrieved_examples["image"][i],
            "similarity_score": round(similarity_score, 2),  # Already in 0-100 range
            "file_name": file_name,
            "metadata": {}
        }
        
        # Add metadata from CSV
        if metadata_df is not None:
            try:
                product_row = None
                if product_id is not None:
                    product_row = metadata_df[metadata_df['id'] == product_id]
                
                # Try to extract ID from filename
                if (product_row is None or product_row.empty) and file_name:
                    import re
                    match = re.search(r'(\d+)', str(file_name))
                    if match:
                        file_id = int(match.group(1))
                        product_row = metadata_df[metadata_df['id'] == file_id]
                
                if product_row is not None and not product_row.empty:
                    row = product_row.iloc[0]
                    product_info["metadata"] = {
                        "productDisplayName": row.get('productDisplayName', 'N/A'),
                        "gender": row.get('gender', 'N/A'),
                        "masterCategory": row.get('masterCategory', 'N/A'),
                        "subCategory": row.get('subCategory', 'N/A'),
                        "articleType": row.get('articleType', 'N/A'),
                        "baseColour": row.get('baseColour', 'N/A'),
                        "season": row.get('season', 'N/A'),
                        "year": row.get('year', 'N/A'),
                        "usage": row.get('usage', 'N/A')
                    }
            except Exception as e:
                pass
        
        results.append(product_info)
    
    return results

# --- UI ---
st.title("🛍️ Visual Product Matcher")
st.markdown("Upload a fashion product image or provide a URL to find similar items!")

# Input method selector
input_method = st.radio("Choose input method:", ["Upload Image", "Image URL"], horizontal=True)

uploaded_file = None
image = None

if input_method == "Upload Image":
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png"],
        help="Upload a product image to find similar items"
    )
    if uploaded_file:
        image = Image.open(uploaded_file)
else:
    image_url = st.text_input(
        "Enter image URL:",
        placeholder="https://example.com/product.jpg",
        help="Paste a direct link to a product image"
    )
    if image_url:
        if is_valid_url(image_url):
            image = load_image_from_url(image_url)
        else:
            st.error("Please enter a valid URL")

# Filters
col1, col2 = st.columns(2)
with col1:
    min_similarity = st.slider(
        "Minimum Similarity (%)",
        min_value=0,
        max_value=100,
        value=0,
        step=5,
        help="Filter results by similarity score"
    )
with col2:
    num_results = st.slider(
        "Number of Results",
        min_value=3,
        max_value=10,
        value=6,
        step=1,
        help="How many similar products to display"
    )

# Display uploaded image
if image:
    st.subheader("📸 Your Image")
    st.image(image, width=300, caption="Input Image")
    
    # Find similar products button
    if st.button("🔍 Find Similar Products", type="primary"):
        with st.spinner("🔄 Searching for similar products..."):
            try:
                # Find similar products
                results = find_similar_products(image, top_k=10)
                
                # Filter by similarity threshold
                filtered_results = [
                    r for r in results 
                    if r["similarity_score"] >= min_similarity
                ][:num_results]
                
                if filtered_results:
                    st.success(f"✅ Found {len(filtered_results)} similar products!")
                    
                    # Display results in grid
                    cols = st.columns(3)
                    for idx, product in enumerate(filtered_results):
                        with cols[idx % 3]:
                            st.image(
                                product["image"],
                                use_container_width=True,
                                caption=f"Match {idx + 1}"
                            )
                            
                            # Similarity score with color
                            score = product["similarity_score"]
                            if score >= 80:
                                color = "🟢"
                            elif score >= 60:
                                color = "🟡"
                            else:
                                color = "🔴"
                            
                            st.markdown(f"**{color} Similarity: {score}%**")
                            
                            # Show metadata if available
                            if product["metadata"]:
                                meta = product["metadata"]
                                if meta.get("productDisplayName", "N/A") != "N/A":
                                    st.markdown(f"**{meta['productDisplayName']}**")
                                
                                with st.expander("📋 View Details"):
                                    st.write(f"**Category:** {meta.get('masterCategory', 'N/A')}")
                                    st.write(f"**Type:** {meta.get('articleType', 'N/A')}")
                                    st.write(f"**Color:** {meta.get('baseColour', 'N/A')}")
                                    st.write(f"**Gender:** {meta.get('gender', 'N/A')}")
                                    st.write(f"**Season:** {meta.get('season', 'N/A')}")
                else:
                    st.warning(f"No products found with similarity >= {min_similarity}%. Try lowering the threshold.")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("🚀 Powered by Vision Transformer (ViT) & FAISS")
