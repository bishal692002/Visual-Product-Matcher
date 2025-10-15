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

# Custom CSS for better UI - Theme-aware styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Reset */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main container */
    .main {
        padding: 3rem 2rem;
        max-width: 1400px;
        margin: 0 auto;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Title styling - Elegant and minimal */
    h1 {
        font-size: 3.5rem !important;
        font-weight: 300 !important;
        letter-spacing: -0.02em;
        text-align: center;
        margin-bottom: 0.5rem !important;
        margin-top: 1rem !important;
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        font-size: 1.125rem;
        font-weight: 400;
        margin-bottom: 3rem;
        line-height: 1.6;
        opacity: 0.8;
    }
    
    /* Section headers */
    h3 {
        font-size: 1.125rem !important;
        font-weight: 600 !important;
        margin-top: 2.5rem !important;
        margin-bottom: 1rem !important;
        letter-spacing: -0.01em;
    }
    
    /* Radio buttons - Minimal tabs style */
    .stRadio > div {
        border-radius: 12px;
        padding: 0.25rem;
        display: inline-flex;
        gap: 0.25rem;
        background-color: var(--background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    
    .stRadio > div > label {
        background: transparent;
        border-radius: 8px;
        padding: 0.625rem 1.25rem;
        font-weight: 500;
        font-size: 0.9375rem;
        transition: all 0.2s ease;
        opacity: 0.7;
    }
    
    .stRadio > div > label:hover {
        opacity: 1;
        background: rgba(128, 128, 128, 0.1);
    }
    
    /* File uploader */
    .stFileUploader {
        border: 2px dashed rgba(128, 128, 128, 0.3);
        border-radius: 16px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: rgba(128, 128, 128, 0.5);
        background: rgba(128, 128, 128, 0.05);
    }
    
    /* Text input */
    .stTextInput > div > div > input {
        border: 1.5px solid rgba(128, 128, 128, 0.3);
        border-radius: 12px;
        padding: 0.75rem 1rem;
        font-size: 0.9375rem;
        transition: all 0.2s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: rgba(128, 128, 128, 0.6);
        box-shadow: 0 0 0 3px rgba(128, 128, 128, 0.1);
    }
    
    /* Sliders */
    .stSlider {
        padding: 1rem 0;
    }
    
    /* Button - Sophisticated design */
    .stButton > button {
        border: none;
        border-radius: 12px;
        padding: 0.875rem 2rem;
        font-size: 1rem;
        font-weight: 500;
        letter-spacing: -0.01em;
        transition: all 0.2s ease;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }
    
    .stButton > button:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        transform: translateY(-1px);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Success/Error messages */
    .stAlert {
        border-radius: 12px;
        border: none;
        padding: 1rem 1.25rem;
        font-size: 0.9375rem;
    }
    
    /* Images */
    .stImage {
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Product grid */
    [data-testid="column"] {
        padding: 0.75rem;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        border-radius: 8px;
        padding: 0.625rem 1rem;
        font-weight: 500;
        font-size: 0.875rem;
        background: rgba(128, 128, 128, 0.1);
    }
    
    /* Divider */
    hr {
        margin: 3rem 0;
        border: none;
        border-top: 1px solid rgba(128, 128, 128, 0.2);
    }
    
    /* Custom badge styling */
    .badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 500;
        font-size: 0.875rem;
        margin: 0.5rem 0;
    }
    
    /* Expander content */
    .streamlit-expanderContent {
        border-radius: 8px;
        background: rgba(128, 128, 128, 0.05);
    }
    
    /* Product name styling */
    .product-name {
        font-weight: 500;
        font-size: 0.9375rem;
        margin: 0.5rem 0;
    }
    
    /* Footer styling */
    .footer-text {
        text-align: center;
        padding: 2rem 0;
        opacity: 0.6;
    }
    
    .footer-text p {
        font-size: 0.875rem;
        font-weight: 400;
    }
    
    .footer-author {
        font-weight: 500;
        opacity: 0.8;
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
# Header - Minimal and elegant
st.markdown("<h1>Visual Product Matcher</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Discover similar products using advanced AI technology</p>", unsafe_allow_html=True)

# Add spacing
st.markdown("<br>", unsafe_allow_html=True)

# Input method selector - Clean tabs
st.markdown("### Input Method")
input_method = st.radio(
    "Choose input method",
    ["Upload Image", "Image URL"],
    horizontal=True,
    label_visibility="collapsed"
)

uploaded_file = None
image = None

if "Upload" in input_method:
    uploaded_file = st.file_uploader(
        "Drop an image file here or click to browse",
        type=["jpg", "jpeg", "png"],
        help="Upload a fashion product image"
    )
    if uploaded_file:
        image = Image.open(uploaded_file)
else:
    image_url = st.text_input(
        "Enter image URL",
        placeholder="https://example.com/image.jpg",
        help="Paste a direct link to an image"
    )
    if image_url:
        if is_valid_url(image_url):
            image = load_image_from_url(image_url)
        else:
            st.error("Please enter a valid URL")

# Filters section
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### Filters")

col1, col2 = st.columns(2)
with col1:
    min_similarity = st.slider(
        "Minimum Similarity",
        min_value=0,
        max_value=100,
        value=0,
        step=5,
        help="Filter results by similarity percentage"
    )
with col2:
    num_results = st.slider(
        "Results to Show",
        min_value=3,
        max_value=10,
        value=6,
        step=1,
        help="Number of similar products to display"
    )

# Display uploaded image
if image:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("### Your Image")
    
    # Center the image
    col1, col2, col3 = st.columns([1.5, 1, 1.5])
    with col2:
        st.image(image, width=320)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Search button
    if st.button("Find Similar Products", type="primary", use_container_width=True):
        with st.spinner("Analyzing image..."):
            try:
                # Find similar products
                results = find_similar_products(image, top_k=10)
                
                # Filter by similarity threshold
                filtered_results = [
                    r for r in results 
                    if r["similarity_score"] >= min_similarity
                ][:num_results]
                
                if filtered_results:
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    st.success(f"Found {len(filtered_results)} similar products")
                    st.markdown("### Similar Products")
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Display results in grid
                    cols = st.columns(3)
                    for idx, product in enumerate(filtered_results):
                        with cols[idx % 3]:
                            st.image(
                                product["image"],
                                use_container_width=True
                            )
                            
                            # Similarity badge - minimal design
                            score = product["similarity_score"]
                            if score >= 80:
                                badge_bg = "#f0fdf4"
                                badge_text = "#166534"
                            elif score >= 60:
                                badge_bg = "#fffbeb"
                                badge_text = "#92400e"
                            else:
                                badge_bg = "#fef2f2"
                                badge_text = "#991b1b"
                            
                            st.markdown(f"""
                                <div style="
                                    background: {badge_bg};
                                    color: {badge_text};
                                    padding: 0.5rem 1rem;
                                    border-radius: 8px;
                                    text-align: center;
                                    margin: 0.75rem 0;
                                    font-weight: 500;
                                    font-size: 0.875rem;
                                ">
                                    {score}% Match
                                </div>
                            """, unsafe_allow_html=True)
                            
                            # Product name
                            if product["metadata"]:
                                meta = product["metadata"]
                                if meta.get("productDisplayName", "N/A") != "N/A":
                                    st.markdown(f"<p class='product-name'>{meta['productDisplayName']}</p>", unsafe_allow_html=True)
                                
                                with st.expander("View Details"):
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

# Footer - Minimal
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
    <div class="footer-text">
        <p>
            Developed by <span class="footer-author">Bishal Biswas</span>
        </p>
    </div>
""", unsafe_allow_html=True)
