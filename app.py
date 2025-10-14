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
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 2rem;
    }
    
    /* Title styling */
    h1 {
        color: #FF6B6B;
        text-align: center;
        font-size: 3rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Card styling for results */
    .product-card {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .product-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0,0,0,0.15);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* File uploader styling */
    .uploadedFile {
        border-radius: 10px;
        border: 2px dashed #667eea;
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background: linear-gradient(to right, #667eea, #764ba2);
    }
    
    /* Success message styling */
    .element-container:has(>.stAlert) {
        border-radius: 10px;
    }
    
    /* Info box styling */
    .info-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

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
# Header with gradient
st.markdown("<h1>🛍️ Visual Product Matcher</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload a fashion product image or provide a URL to find similar items using AI!</p>", unsafe_allow_html=True)

# Add spacing
st.markdown("<br>", unsafe_allow_html=True)

# Input method selector with better styling
st.markdown("### 📤 Choose Your Input Method")
input_method = st.radio(
    "",
    ["📁 Upload Image", "🔗 Image URL"],
    horizontal=True,
    label_visibility="collapsed"
)

uploaded_file = None
image = None

if "📁" in input_method:
    st.markdown("<div class='info-box'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "📸 Choose an image file",
        type=["jpg", "jpeg", "png"],
        help="Upload a product image to find similar items"
    )
    st.markdown("</div>", unsafe_allow_html=True)
    if uploaded_file:
        image = Image.open(uploaded_file)
else:
    st.markdown("<div class='info-box'>", unsafe_allow_html=True)
    image_url = st.text_input(
        "🌐 Enter image URL:",
        placeholder="https://example.com/product.jpg",
        help="Paste a direct link to a product image"
    )
    st.markdown("</div>", unsafe_allow_html=True)
    if image_url:
        if is_valid_url(image_url):
            image = load_image_from_url(image_url)
        else:
            st.error("Please enter a valid URL")

# Filters section with better design
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### ⚙️ Filter Settings")

col1, col2 = st.columns(2)
with col1:
    min_similarity = st.slider(
        "🎯 Minimum Similarity (%)",
        min_value=0,
        max_value=100,
        value=0,
        step=5,
        help="Filter results by similarity score"
    )
with col2:
    num_results = st.slider(
        "📊 Number of Results",
        min_value=3,
        max_value=10,
        value=6,
        step=1,
        help="How many similar products to display"
    )

# Display uploaded image
if image:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📸 Your Input Image")
    
    # Center the image
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, use_container_width=True, caption="Input Image")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Find similar products button with custom styling
    if st.button("🔍 Find Similar Products", type="primary", use_container_width=True):
        with st.spinner("🔄 Analyzing image and searching for similar products..."):
            try:
                # Find similar products
                results = find_similar_products(image, top_k=10)
                
                # Filter by similarity threshold
                filtered_results = [
                    r for r in results 
                    if r["similarity_score"] >= min_similarity
                ][:num_results]
                
                if filtered_results:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.success(f"✨ Found {len(filtered_results)} similar products!")
                    st.markdown("### 🎯 Similar Products")
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Display results in grid
                    cols = st.columns(3)
                    for idx, product in enumerate(filtered_results):
                        with cols[idx % 3]:
                            st.image(
                                product["image"],
                                use_container_width=True
                            )
                            
                            # Similarity score with gradient badge
                            score = product["similarity_score"]
                            if score >= 80:
                                badge_color = "#10b981"  # Green
                                emoji = "🟢"
                            elif score >= 60:
                                badge_color = "#f59e0b"  # Yellow
                                emoji = "🟡"
                            else:
                                badge_color = "#ef4444"  # Red
                                emoji = "🔴"
                            
                            st.markdown(f"""
                                <div style="
                                    background: linear-gradient(135deg, {badge_color}22 0%, {badge_color}44 100%);
                                    padding: 0.5rem;
                                    border-radius: 10px;
                                    text-align: center;
                                    margin: 0.5rem 0;
                                    border-left: 4px solid {badge_color};
                                ">
                                    <strong>{emoji} Match {idx + 1}: {score}%</strong>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            # Show metadata if available
                            if product["metadata"]:
                                meta = product["metadata"]
                                if meta.get("productDisplayName", "N/A") != "N/A":
                                    st.markdown(f"**🏷️ {meta['productDisplayName']}**")
                                
                                with st.expander("📋 View Full Details"):
                                    st.markdown(f"**Category:** {meta.get('masterCategory', 'N/A')}")
                                    st.markdown(f"**Type:** {meta.get('articleType', 'N/A')}")
                                    st.markdown(f"**Color:** {meta.get('baseColour', 'N/A')}")
                                    st.markdown(f"**Gender:** {meta.get('gender', 'N/A')}")
                                    st.markdown(f"**Season:** {meta.get('season', 'N/A')}")
                            
                            st.markdown("<br>", unsafe_allow_html=True)
                else:
                    st.warning(f"No products found with similarity >= {min_similarity}%. Try lowering the threshold.")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Footer with enhanced styling
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
    <div style="text-align: center; padding: 2rem 0; color: #666;">
        <h4>🚀 Powered by AI</h4>
        <p>Vision Transformer (ViT) • FAISS • HuggingFace • Streamlit</p>
        <p style="font-size: 0.9rem; margin-top: 1rem;">
            Built with ❤️ for intelligent product discovery
        </p>
    </div>
""", unsafe_allow_html=True)
