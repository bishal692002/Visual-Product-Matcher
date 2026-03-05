import os
import streamlit as st
from PIL import Image
import requests
import io
import numpy as np
from urllib.parse import urlparse
import pandas as pd
import torch

# Import ML libraries
from datasets import load_dataset, load_from_disk
from transformers import CLIPProcessor, CLIPModel

# Local cache path for the 44K-image index built by build_local_index.py
LOCAL_INDEX_PATH = os.path.join(os.path.dirname(__file__), "cache", "fashion_index")

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

# --- Model + Dataset Loading ---
@st.cache_resource
def load_model_and_dataset():
    """Load FashionCLIP model and fashion image index (local 44K cache or HF fallback)"""
    # ------------------------------------------------------------------ model
    # Architecture: Input Image → FashionCLIP ViT-B/32 Vision Encoder
    #               → 768-Dimensional Pooler Output (L2-normed)
    #               → FAISS Inner-Product Index → Top-K Results
    #
    # FashionCLIP is fine-tuned on ~700K fashion image-text pairs.
    # We extract the raw 768-dim ViT CLS token (before CLIP's 512-dim projection)
    # for richer colour, texture, and garment-shape features.
    MODEL_CKPT = "patrickjohncyh/fashion-clip"
    processor = CLIPProcessor.from_pretrained(MODEL_CKPT)
    clip_model = CLIPModel.from_pretrained(MODEL_CKPT)
    clip_model.eval()
    # Apple Silicon MPS > CUDA > CPU
    _device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    clip_model = clip_model.to(_device)

    # ---------------------------------------------------------------- dataset
    if os.path.exists(LOCAL_INDEX_PATH):
        # Full 44K local index built by build_local_index.py
        dataset = load_from_disk(LOCAL_INDEX_PATH)
        dataset_size = len(dataset)
    else:
        # Load full 44K embeddings dataset from HuggingFace Hub
        DATASET_REPO = "bishal692002/fashion-products-embeddings"
        dataset = load_dataset(DATASET_REPO, split="train")
        dataset_size = len(dataset)

    # Embeddings in the local index are already L2-normalised.
    # Embeddings in the old HF dataset may not be — normalise to be safe.
    raw_embs = np.array(dataset["embeddings"]).astype("float32")
    norms = np.linalg.norm(raw_embs, axis=1, keepdims=True)
    normed = raw_embs / np.maximum(norms, 1e-8)
    # Only add the column if it doesn't already exist
    if "embeddings_norm" not in dataset.column_names:
        dataset = dataset.add_column("embeddings_norm", normed.tolist())
    # Build cosine-similarity FAISS index (inner-product on unit vectors)
    dataset.add_faiss_index(column="embeddings_norm", metric_type=0)

    # ---------------------------------------------------------------- metadata
    try:
        metadata_df = pd.read_csv(
            os.path.join(os.path.dirname(__file__), "img", "styles.csv"),
            on_bad_lines="skip"
        )
    except Exception:
        metadata_df = None

    return processor, clip_model, dataset, metadata_df, dataset_size

# Load everything
processor, clip_model, dataset, metadata_df, dataset_size = load_model_and_dataset()

# --- Helper Functions ---
def extract_embeddings(image: Image.Image) -> np.ndarray:
    """
    Extract an L2-normalised 768-dim image embedding using FashionCLIP's ViT-B/32.

    Architecture step: Image Preprocessing → Vision Transformer → 768-Dimensional Vector

    We take the raw ViT pooler output (768-dim CLS token) instead of going through
    CLIP's visual_projection (which squeezes to 512-dim). The 768-dim representation
    retains more colour, texture, and shape information for image-to-image retrieval.
    """
    _device = next(clip_model.parameters()).device
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(_device)
    with torch.no_grad():
        vision_out = clip_model.vision_model(pixel_values=pixel_values)
        features = vision_out.pooler_output  # (1, 768) — raw ViT CLS token
    vec = features.squeeze().cpu().numpy()  # always back to CPU/numpy
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

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

def find_similar_products(query_image, top_k=10, text_hint: str = ""):
    """
    Find similar products using the 768-dim FashionCLIP FAISS index.

    Pipeline (matches system architecture diagram):
      1. Image Preprocessing  — resize & normalise via CLIPProcessor
      2. Vision Transformer   — FashionCLIP ViT-B/32 pooler output
      3. 768-Dimensional Vector — L2-normalised CLS token
      4. FAISS Inner-Product Index — cosine similarity search (inner product on unit vectors)
      5. Top-K Similar Products + Similarity Scores

    Text Hint (optional):
      If the user provides a description (e.g. "green polo t-shirt"), results are
      re-ranked AFTER FAISS retrieval by boosting items whose metadata keywords
      (colour, articleType) match the hint. This avoids the incompatible-space
      issue of blending 768-dim image vectors with CLIP's 512-dim text vectors.
    """
    query_embedding = extract_embeddings(query_image)  # 768-dim L2-normed vector

    # Fetch 6× candidates for high recall — FAISS sorts by cosine similarity
    fetch_k = min(top_k * 6, len(dataset))
    scores, retrieved = dataset.get_nearest_examples(
        "embeddings_norm", query_embedding, k=fetch_k
    )

    # --- Score rescaling ---
    # 768-dim ViT pooler cosine scores sit in ~[0.50, 0.98] for fashion images.
    # Rescale to [0, 100] so top matches show 85–98%, unrelated items show 0–20%.
    LOW  = 0.50   # cosine score → 0 %
    HIGH = 0.98   # cosine score → 100 %

    def rescale(raw_score):
        pct = (float(raw_score) - LOW) / (HIGH - LOW) * 100.0
        return round(max(0.0, min(100.0, pct)), 1)

    def _get(col, i):
        col_data = retrieved.get(col)
        if col_data is None:
            return "N/A"
        val = col_data[i]
        return str(val) if val is not None and str(val) not in ("", "nan", "None") else "N/A"

    results = []
    for i in range(len(retrieved["image"])):
        raw   = float(scores[i])
        score = rescale(raw)
        product_info = {
            "image": retrieved["image"][i],
            "similarity_score": score,
            "raw_score": raw,
            "metadata": {
                "productDisplayName": _get("productDisplayName", i),
                "gender":             _get("gender", i),
                "masterCategory":     _get("masterCategory", i),
                "subCategory":        _get("subCategory", i),
                "articleType":        _get("articleType", i),
                "baseColour":         _get("baseColour", i),
                "season":             _get("season", i),
                "year":               _get("year", i),
                "usage":              _get("usage", i),
            }
        }
        results.append(product_info)

    # --- Text-hint re-ranking (post-search metadata boosting) ---
    # We work in metadata-space so there is no embedding-space mismatch.
    if text_hint.strip():
        hint_words = set(text_hint.lower().split())

        def hint_bonus(item):
            meta = item["metadata"]
            fields = " ".join([
                meta.get("baseColour", ""),
                meta.get("articleType", ""),
                meta.get("subCategory", ""),
                meta.get("productDisplayName", ""),
                meta.get("gender", ""),
            ]).lower()
            # Count how many hint words appear in the metadata
            matches = sum(1 for w in hint_words if w in fields)
            return matches

        # Sort: primarily by hint word matches, secondarily by cosine score
        results.sort(key=lambda x: (hint_bonus(x), x["raw_score"]), reverse=True)
    else:
        results.sort(key=lambda x: x["raw_score"], reverse=True)

    return results[:top_k]

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
        min_value=4,
        max_value=12,
        value=8,
        step=4,
        help="Number of similar products to display"
    )

text_hint = st.text_input(
    "Describe the product (optional — boosts matching items to the top)",
    placeholder="e.g. green polo t-shirt, black running shoes, floral dress…",
    help="Keywords are matched against product colour, type, and name to re-rank results after visual search."
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
        with st.spinner("Analyzing image…"):
            try:
                # Find similar products (pass optional text hint for guided search)
                results = find_similar_products(image, top_k=num_results, text_hint=text_hint)
                
                # Filter by similarity threshold
                filtered_results = [
                    r for r in results 
                    if r["similarity_score"] >= min_similarity
                ][:num_results]
                
                if filtered_results:
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    hint_note = f" (guided by: \"{text_hint}\")" if text_hint.strip() else ""
                    st.success(f"Found {len(filtered_results)} similar products — searched {dataset_size:,} items{hint_note}")
                    st.markdown("### Similar Products")
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Display results in 4-column grid with fixed image size
                    cols = st.columns(4)
                    for idx, product in enumerate(filtered_results):
                        with cols[idx % 4]:
                            # Resize for uniform card display
                            thumb = product["image"].copy()
                            thumb.thumbnail((280, 320), Image.LANCZOS)
                            st.image(thumb, width=240)
                            
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
