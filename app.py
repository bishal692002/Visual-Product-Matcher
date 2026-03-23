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
        # Load embeddings dataset from HuggingFace Hub (configurable for deployment)
        DATASET_REPO = os.getenv("EMBEDDINGS_DATASET_REPO", "bishal692002/fashion-products-embeddings")
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
    # IMPORTANT: Always rebuild the index, don't rely on cached index
    # (HuggingFace datasets don't persist FAISS indices when saved to disk)
    try:
        dataset.add_faiss_index(column="embeddings_norm", metric_type=0)
    except Exception as e:
        st.warning(f"⚠️ Failed to build FAISS index: {e}")
        raise

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
    Extract an L2-normalised 768-dim image embedding using FashionCLIP's ViT pooler.
    
    Uses CLIP's raw vision_model pooler output (768-dim) — NOT the visual_projection.
    This matches the embeddings stored in the HuggingFace dataset built by build_local_index.py.
    
    Why 768-dim?
    - ViT-B/32 produces 768-dim CLS token (full semantic richness)
    - visual_projection compresses to 512-dim for image-text joint space
    - For image-only search, we want the richer representation
    """
    _device = next(clip_model.parameters()).device
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(_device)
    with torch.no_grad():
        vision_out = clip_model.vision_model(pixel_values=pixel_values)
        features = vision_out.pooler_output  # 768-dim (NOT visual_projection)
    vec = features.squeeze().cpu().numpy()
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

def resolve_result_image(product: dict):
    """Resolve a result image from PIL, HF image dict, URL/path metadata, or return None."""
    img = product.get("image")
    if isinstance(img, Image.Image):
        return img

    if isinstance(img, dict):
        if img.get("bytes"):
            try:
                return Image.open(io.BytesIO(img["bytes"])).convert("RGB")
            except Exception:
                pass
        if img.get("path"):
            try:
                return Image.open(img["path"]).convert("RGB")
            except Exception:
                pass

    metadata = product.get("metadata", {}) or {}
    image_url = metadata.get("image_url")
    if image_url and image_url != "N/A":
        return load_image_from_url(image_url)

    image_path = metadata.get("image_path")
    if image_path and image_path != "N/A":
        try:
            return Image.open(image_path).convert("RGB")
        except Exception:
            return None

    return None

def prepare_display_image(image: Image.Image, min_width: int = 380) -> Image.Image:
    """Upscale small catalog thumbnails so result cards show prominent product visuals."""
    if not isinstance(image, Image.Image):
        return image

    img = image.convert("RGB")
    if img.width >= min_width:
        return img

    scale = min_width / max(img.width, 1)
    new_height = int(img.height * scale)
    return img.resize((min_width, max(new_height, 1)), Image.LANCZOS)

def make_prominent_product_image(image: Image.Image, canvas_width: int = 520, canvas_height: int = 640) -> Image.Image:
    """Crop background margins and place product on a large canvas for strong visual presence."""
    if not isinstance(image, Image.Image):
        return image

    img = image.convert("RGB")
    arr = np.array(img).astype(np.int16)

    # Estimate the plain background color from corners, then keep pixels that differ.
    h, w, _ = arr.shape
    corners = np.array([
        arr[0, 0],
        arr[0, w - 1],
        arr[h - 1, 0],
        arr[h - 1, w - 1],
    ])
    bg = np.median(corners, axis=0)
    dist = np.sqrt(((arr - bg) ** 2).sum(axis=2))
    mask = dist > 18
    ys, xs = np.where(mask)

    if len(xs) > 0 and len(ys) > 0:
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        pad = 6
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(img.width, x1 + pad)
        y1 = min(img.height, y1 + pad)
        img = img.crop((x0, y0, x1, y1))

    # Upscale object region to fill most of a tall product-card canvas
    scale = min((canvas_width * 0.92) / max(img.width, 1), (canvas_height * 0.78) / max(img.height, 1))
    new_w = max(1, int(img.width * scale))
    new_h = max(1, int(img.height * scale))
    img = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
    x = (canvas_width - new_w) // 2
    y = max(12, (canvas_height - new_h) // 2)
    canvas.paste(img, (x, y))
    return canvas

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

    image_count = len(retrieved.get("image", []))
    score_count = len(scores)
    safe_count = min(image_count, score_count)

    results = []
    for i in range(safe_count):
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
                "image_url":          _get("image_url", i),
                "image_path":         _get("image_path", i),
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

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
# Single Item Search URL state
if "single_item_url" not in st.session_state:
    st.session_state.single_item_url = ""
if "single_item_preview" not in st.session_state:
    st.session_state.single_item_preview = None
if "single_item_url_error" not in st.session_state:
    st.session_state.single_item_url_error = None

# Full Outfit Analysis URL state
if "outfit_url" not in st.session_state:
    st.session_state.outfit_url = ""
if "outfit_preview" not in st.session_state:
    st.session_state.outfit_preview = None
if "outfit_url_error" not in st.session_state:
    st.session_state.outfit_url_error = None

# Debug info (optional, can be removed later)
with st.sidebar:
    with st.expander("🔧 Debug Info"):
        st.write("**Session State:**")
        st.text(f"single_item_url: {st.session_state.single_item_url}")
        st.text(f"single_item_preview: {'Image loaded' if st.session_state.single_item_preview else 'None'}")
        st.text(f"single_item_url_error: {st.session_state.single_item_url_error}")
        st.text(f"outfit_url: {st.session_state.outfit_url}")
        st.text(f"outfit_preview: {'Image loaded' if st.session_state.outfit_preview else 'None'}")
        st.text(f"outfit_url_error: {st.session_state.outfit_url_error}")


# ============================================================================
# CALLBACK FUNCTIONS FOR URL PREVIEW
# ============================================================================
def load_single_item_url_preview():
    """Callback: Load preview when URL input changes (on Enter)"""
    try:
        url = st.session_state.get("single_item_url_input", "").strip()
        st.session_state.single_item_url = url
        st.session_state.single_item_url_error = None
        st.session_state.single_item_preview = None
        
        if not url:
            return
        
        if not is_valid_url(url):
            st.session_state.single_item_url_error = "Invalid URL format. Please enter a valid image URL."
            return
        
        # Try to load the image
        image = load_image_from_url(url)
        if image is None:
            st.session_state.single_item_url_error = "Unable to load image from URL. Please check the link and try again."
        else:
            st.session_state.single_item_preview = image
            st.session_state.single_item_url_error = None
    except Exception as e:
        st.session_state.single_item_url_error = f"Error processing URL: {str(e)}"

def load_outfit_url_preview():
    """Callback: Load preview when URL input changes (on Enter)"""
    try:
        url = st.session_state.get("outfit_url_input", "").strip()
        st.session_state.outfit_url = url
        st.session_state.outfit_url_error = None
        st.session_state.outfit_preview = None
        
        if not url:
            return
        
        if not is_valid_url(url):
            st.session_state.outfit_url_error = "Invalid URL format. Please enter a valid image URL."
            return
        
        # Try to load the image
        image = load_image_from_url(url)
        if image is None:
            st.session_state.outfit_url_error = "Unable to load image from URL. Please check the link and try again."
        else:
            st.session_state.outfit_preview = image
            st.session_state.outfit_url_error = None
    except Exception as e:
        st.session_state.outfit_url_error = f"Error processing URL: {str(e)}"

# ============================================================================
# MODE SELECTOR - Choose between Single Item or Full Outfit Analysis
# ============================================================================
analysis_mode = st.radio(
    "Select analysis mode",
    ["Single Item Search", "Full Outfit Analysis"],
    horizontal=True,
    label_visibility="collapsed"
)

# ============================================================================
# MODE 1: SINGLE ITEM SEARCH (Existing functionality)
# ============================================================================
if analysis_mode == "Single Item Search":
    st.markdown("---")
    
    # Input method selector
    input_method = st.radio(
        "How would you like to search?",
        ["Upload Image", "Image URL"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    st.markdown("")  # Small spacer

    uploaded_file = None
    image = None

    if "Upload" in input_method:
        uploaded_file = st.file_uploader(
            "Upload your product image",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear fashion product image"
        )
        if uploaded_file:
            image = Image.open(uploaded_file)
            # Clear URL state when switching to upload
            st.session_state.single_item_url = ""
            st.session_state.single_item_preview = None
            st.session_state.single_item_url_error = None
    else:
        # URL input with on_change callback for instant preview
        st.text_input(
            "Paste image URL",
            placeholder="https://example.com/image.jpg",
            help="Paste a direct link and press Enter to preview",
            key="single_item_url_input",
            on_change=load_single_item_url_preview
        )
        
        # Display URL error if present
        error_msg = st.session_state.get("single_item_url_error")
        if error_msg:
            st.error(error_msg)
        
        # Use preview image from session state
        image = st.session_state.get("single_item_preview")
    
    st.markdown("")  # Small spacer
    
    # Text hint for guided search
    text_hint = st.text_input(
        "Product description (optional)",
        placeholder="e.g. green polo shirt, black leather shoes…",
        help="Add optional details to guide the search results"
    )

    # Display uploaded image and search button
    if image:
        st.markdown("")
        col_preview, col_button = st.columns([1.5, 2])

        with col_preview:
            st.markdown("**Your product:**")
            st.image(image, width=300, use_container_width=False)

        with col_button:
            st.markdown("")
            st.markdown("**Search Settings:**")
            min_similarity = st.slider(
                "Min Similarity %",
                min_value=0,
                max_value=100,
                value=0,
                step=5,
                label_visibility="collapsed",
            )
            num_results = st.slider(
                "Results to show",
                min_value=4,
                max_value=12,
                value=8,
                step=4,
                label_visibility="collapsed",
            )
            st.markdown("")
            if st.button("🔍 Find Similar Products", type="primary", use_container_width=True):
                with st.spinner("Analyzing image…"):
                    try:
                        results = find_similar_products(image, top_k=num_results, text_hint=text_hint)
                        filtered_results = [
                            r for r in results if r["similarity_score"] >= min_similarity
                        ][:num_results]

                        if filtered_results:
                            hint_note = f" (guided by: \"{text_hint}\")" if text_hint.strip() else ""
                            st.success(f"Found {len(filtered_results)} similar products{hint_note}")
                            st.markdown("### Top Matches")
                            st.markdown("")

                            cols = st.columns(2, gap="large")
                            for idx, product in enumerate(filtered_results):
                                with cols[idx % 2]:
                                    with st.container(border=True):
                                        st.markdown("<div style='padding: 0.5rem 0.75rem 1rem 0.75rem;'>", unsafe_allow_html=True)
                                        meta = product.get("metadata", {}) or {}
                                        score = product.get("similarity_score", 0)

                                        # [Image]
                                        resolved_image = resolve_result_image(product)
                                        if resolved_image is not None:
                                            display_image = (
                                                resolved_image.copy()
                                                if hasattr(resolved_image, "copy")
                                                else resolved_image
                                            )
                                            if isinstance(display_image, Image.Image):
                                                display_image = prepare_display_image(display_image, min_width=420)
                                                display_image = make_prominent_product_image(display_image, canvas_width=520, canvas_height=640)
                                                st.image(display_image, width=460)
                                            else:
                                                st.info("No image available")
                                        else:
                                            st.info("No image available")

                                        # [Match %]
                                        if score >= 80:
                                            badge_bg = "#10b981"
                                            badge_text = "white"
                                        elif score >= 60:
                                            badge_bg = "#f59e0b"
                                            badge_text = "white"
                                        else:
                                            badge_bg = "#ef4444"
                                            badge_text = "white"

                                        st.markdown(
                                            f"""
                                            <div style="
                                                background: {badge_bg};
                                                color: {badge_text};
                                                padding: 0.6rem 1rem;
                                                border-radius: 6px;
                                                text-align: center;
                                                margin: 0.75rem 0;
                                                font-weight: 600;
                                                font-size: 0.95rem;
                                            ">
                                                {score}% Match
                                            </div>
                                            """,
                                            unsafe_allow_html=True,
                                        )

                                        # [Title]
                                        name = meta.get("productDisplayName", "Unknown Product")
                                        st.markdown(
                                            f"<p style='font-weight: 600; font-size: 0.9rem; margin: 0.75rem 0 0.5rem 0; line-height: 1.3;'>{name[:60]}</p>",
                                            unsafe_allow_html=True,
                                        )

                                        # [Metadata]
                                        meta_info = []
                                        if meta.get("masterCategory") and meta["masterCategory"] != "N/A":
                                            meta_info.append(meta["masterCategory"])
                                        if meta.get("articleType") and meta["articleType"] != "N/A":
                                            meta_info.append(meta["articleType"])
                                        if meta.get("baseColour") and meta["baseColour"] != "N/A":
                                            meta_info.append(meta["baseColour"])

                                        if meta_info:
                                            st.markdown(
                                                f"<div style='font-size: 0.8rem; line-height: 1.5; color: #666; margin-top: 0.5rem;'>{' • '.join(meta_info)}</div>",
                                                unsafe_allow_html=True,
                                            )
                                        st.markdown("</div>", unsafe_allow_html=True)
                        else:
                            st.warning(
                                f"No products found with similarity >= {min_similarity}%. Try lowering the threshold."
                            )
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

# ============================================================================
# MODE 2: FULL OUTFIT ANALYSIS (New functionality)
# ============================================================================
elif analysis_mode == "Full Outfit Analysis":
    st.markdown("---")
    
    # Import outfit analysis service (lazy-loaded, only when this mode is selected)
    from services.outfit_search import OutfitSearchService
    
    # Input method selector
    outfit_input_method = st.radio(
        "How would you like to upload your outfit?",
        ["Upload Image", "Image URL"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    st.markdown("")  # Small spacer
    
    outfit_image = None
    
    if "Upload" in outfit_input_method:
        outfit_file = st.file_uploader(
            "Upload your outfit image",
            type=["jpg", "jpeg", "png"],
            key="outfit_uploader",
            help="A clear full-body or styled outfit image works best"
        )
        if outfit_file:
            outfit_image = Image.open(outfit_file)
            # Clear URL state when switching to upload
            st.session_state.outfit_url = ""
            st.session_state.outfit_preview = None
            st.session_state.outfit_url_error = None
    else:
        # URL input with on_change callback for instant preview
        st.text_input(
            "Paste outfit image URL",
            placeholder="https://example.com/outfit.jpg",
            help="Paste a direct link and press Enter to preview",
            key="outfit_url_input",
            on_change=load_outfit_url_preview
        )
        
        # Display URL error if present
        error_msg = st.session_state.get("outfit_url_error")
        if error_msg:
            st.error(error_msg)
        
        # Use preview image from session state
        outfit_image = st.session_state.get("outfit_preview")
    
    if outfit_image:
        st.markdown("")
        
        # Display input image
        col1, col2 = st.columns([1.5, 2])
        with col1:
            st.markdown("**Your outfit:**")
            st.image(outfit_image, width=300, use_container_width=False)
        
        with col2:
            st.markdown("")
            st.markdown("**Analysis Options:**")
            top_k = st.slider(
                "Items to show per category",
                min_value=3,
                max_value=12,
                value=6,
                step=1,
                label_visibility="collapsed"
            )
            st.markdown("")
            if st.button("🔍 Analyze Outfit", type="primary", use_container_width=True):
                with st.spinner("Detecting clothing items and finding similar products..."):
                    try:
                        # Initialize outfit search service
                        outfit_service = OutfitSearchService(processor, clip_model, dataset, metadata_df)
                        
                        # Run full analysis
                        analysis = outfit_service.analyze_outfit(outfit_image, top_k_per_item=top_k)
                        
                        if analysis.get("error"):
                            st.error(f"❌ {analysis['error']}")
                        else:
                            # Display results
                            detected_items = analysis.get("detected_items", [])
                            num_items = len(detected_items)
                            
                            if num_items == 0:
                                st.warning("No clothing items detected. Please try a clearer outfit image.")
                            else:
                                st.success(f"✨ Detected {num_items} item(s)")
                                
                                # Show detected items with boxes
                                st.markdown("<br>", unsafe_allow_html=True)
                                st.markdown("### Detected Items")
                                
                                col_left, col_right = st.columns([1, 2])
                                
                                with col_left:
                                    if analysis.get("outfit_image_with_boxes"):
                                        st.markdown("**Outfit with Detections:**")
                                        st.image(analysis["outfit_image_with_boxes"], width=250)
                                
                                with col_right:
                                    st.markdown("**Items Found:**")
                                    for item in detected_items:
                                        category = item["category"].capitalize()
                                        confidence = item["confidence"]
                                        st.write(f"✓ **{category}** (Confidence: {confidence:.0%})")
                                
                                # Show results per item in tabs
                                st.markdown("<br><br>", unsafe_allow_html=True)
                                st.markdown("### Similar Items Per Category")
                                
                                # Create tabs for each detected item
                                tab_labels = [f"{item['category'].capitalize()} (Item {item['item_id']})" 
                                             for item in detected_items]
                                tabs = st.tabs(tab_labels)
                                
                                for tab, item in zip(tabs, detected_items):
                                    with tab:
                                        col_crop, col_results = st.columns([1, 4])
                                        
                                        # Show cropped item
                                        with col_crop:
                                            st.markdown("**Detected Item:**")
                                            st.image(item["cropped_image"], width=150)
                                        
                                        # Show search results  
                                        with col_results:
                                            st.markdown("**Top Matches:**")
                                            search_results = item.get("search_results", [])
                                            
                                            if search_results:
                                                # Display in 2-column grid for larger product cards
                                                result_cols = st.columns(2, gap="large")
                                                for idx, result in enumerate(search_results[:6]):
                                                    with result_cols[idx % 2]:
                                                        with st.container(border=True):
                                                            # Image
                                                            result_image = resolve_result_image(result)
                                                            if result_image is not None:
                                                                result_image = prepare_display_image(result_image, min_width=340)
                                                                result_image = make_prominent_product_image(result_image, canvas_width=420, canvas_height=520)
                                                                st.image(result_image, width=360)
                                                            else:
                                                                st.info("No image available")

                                                            # Score badge
                                                            score = result["similarity_score"]
                                                            if score >= 80:
                                                                badge_color = "#10b981"  # green
                                                            elif score >= 60:
                                                                badge_color = "#f59e0b"  # amber
                                                            else:
                                                                badge_color = "#ef4444"  # red

                                                            st.markdown(f"""
<div style="
    background: {badge_color}20;
    color: {badge_color};
    padding: 0.5rem;
    border-radius: 8px;
    text-align: center;
    margin: 0.5rem 0;
    font-weight: 600;
    font-size: 0.875rem;
">
    {score}% Match
</div>
                                                            """, unsafe_allow_html=True)

                                                            # Product name
                                                            st.caption(result["metadata"].get("productDisplayName", "Unknown")[:50])
                                            else:
                                                st.info("No matches found for this item")
                                
                                # Outfit Recommendations
                                st.markdown("<br><br>", unsafe_allow_html=True)
                                st.markdown("### 👕 Complete the Look")
                                
                                recommendations = analysis.get("outfit_recommendations", {})
                                comp_items = recommendations.get("complementary_items", [])
                                
                                if comp_items:
                                    for rec in comp_items:
                                        detected_cat = rec.get("detected_category", "item").upper()
                                        st.markdown(f"**Based on your {detected_cat}:**")
                                        
                                        recs_list = rec.get("recommendations", [])
                                        for comp in recs_list:
                                            comp_cat = comp.get("category", "").upper()
                                            comp_items_list = comp.get("items", [])
                                            
                                            if comp_items_list:
                                                st.markdown(f"*Recommended {comp_cat}:*")
                                                for sugg_item in comp_items_list[:3]:
                                                    st.write(
                                                        f"- {sugg_item['product_name']} "
                                                        f"({sugg_item['color']}) — {sugg_item['reason']}"
                                                    )
                                        st.markdown("<br>", unsafe_allow_html=True)
                                
                                # Analysis time
                                analysis_time = analysis.get("analysis_time_ms", 0)
                                st.caption(f"⏱️ Analysis completed in {analysis_time:.0f}ms")
                    
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {str(e)}")

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
