import streamlit as st
from PIL import Image
import requests
import io
import base64
from urllib.parse import urlparse

# --- Page Configuration ---
st.set_page_config(
    page_title="Visual Product Recommender",
    page_icon="🛍️",
    layout="wide"
)

# --- FastAPI Backend URL ---
# This is the address where your FastAPI backend is running.
# If you are running both frontend and backend on the same machine, this is correct.
BACKEND_URL = "http://127.0.0.1:8000/recommend/"

# --- Helper Functions ---
def is_valid_url(url):
    """Check if a string is a valid URL"""
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

# --- UI Components ---
st.title("🛍️ Visual Product Recommender")
st.markdown(
    "Upload an image or provide a URL of a fashion product, and we'll show you similar items from our catalog!"
)

# Add input method selector
input_method = st.radio("Choose input method:", ["Upload Image", "Image URL"], horizontal=True)

uploaded_file = None
image = None
image_bytes = None

if input_method == "Upload Image":
    # File uploader allows user to upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_bytes = uploaded_file.getvalue()
else:
    # URL input
    image_url = st.text_input("Enter image URL:", placeholder="https://example.com/image.jpg")
    if image_url and is_valid_url(image_url):
        image = load_image_from_url(image_url)
        if image:
            # Convert PIL image to bytes
            buf = io.BytesIO()
            image.save(buf, format='JPEG')
            image_bytes = buf.getvalue()
    elif image_url:
        st.error("Please enter a valid URL")

if image is not None:
    # Display the uploaded/loaded image
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption='Your Uploaded Image', width=300)

    st.divider()
    
    # Add similarity filter slider
    st.subheader("Filter Options")
    min_similarity = st.slider(
        "Minimum Similarity Score (%)",
        min_value=0,
        max_value=100,
        value=0,
        step=5,
        help="Filter results to show only products above this similarity threshold"
    )
    
    # Number of results to display
    num_results = st.slider(
        "Number of Results",
        min_value=3,
        max_value=10,
        value=6,
        step=1
    )

    # When the user clicks the button, send the image to the backend
    if st.button('Find Similar Products', use_container_width=True, type="primary"):
        with st.spinner('AI is searching for recommendations...'):
            try:
                # Prepare the image file to be sent in the request
                files = {'file': ('image.jpg', image_bytes, 'image/jpeg')}
                
                # Make a POST request to the FastAPI backend
                response = requests.post(BACKEND_URL, files=files, timeout=30)
                
                # Check if the request was successful
                if response.status_code == 200:
                    data = response.json()
                    recommendations = data.get("recommendations", [])
                    
                    # Filter recommendations by similarity score
                    filtered_recommendations = [
                        rec for rec in recommendations 
                        if rec.get("similarity_score", 0) >= min_similarity
                    ]
                    
                    # Limit to requested number of results
                    filtered_recommendations = filtered_recommendations[:num_results]
                    
                    if len(filtered_recommendations) == 0:
                        st.warning(f"No products found with similarity score >= {min_similarity}%. Try lowering the threshold.")
                    else:
                        st.success(f"Found {len(filtered_recommendations)} similar products!")
                        
                        # Display the recommended images in a grid
                        # Use columns based on number of results
                        cols_per_row = min(3, len(filtered_recommendations))
                        
                        for idx in range(0, len(filtered_recommendations), cols_per_row):
                            cols = st.columns(cols_per_row)
                            for col_idx, col in enumerate(cols):
                                rec_idx = idx + col_idx
                                if rec_idx < len(filtered_recommendations):
                                    rec = filtered_recommendations[rec_idx]
                                    
                                    with col:
                                        # Decode the Base64 string back into an image
                                        rec_image = Image.open(io.BytesIO(base64.b64decode(rec["image"])))
                                        st.image(rec_image, use_container_width=True)
                                        
                                        # Display similarity score
                                        st.metric("Similarity", f"{rec.get('similarity_score', 0)}%")
                                        
                                        # Display product metadata if available
                                        if "product_name" in rec:
                                            st.caption(f"**{rec['product_name']}**")
                                        
                                        with st.expander("Product Details"):
                                            if "category" in rec:
                                                st.write(f"**Category:** {rec['category']}")
                                            if "sub_category" in rec:
                                                st.write(f"**Type:** {rec['sub_category']}")
                                            if "article_type" in rec:
                                                st.write(f"**Article:** {rec['article_type']}")
                                            if "color" in rec:
                                                st.write(f"**Color:** {rec['color']}")
                                            if "gender" in rec:
                                                st.write(f"**Gender:** {rec['gender']}")
                else:
                    st.error(f"Error: Could not get recommendations. Server responded with status {response.status_code}")
                    st.error(f"Details: {response.text}")

            except requests.exceptions.RequestException as e:
                st.error(f"Could not connect to the backend. Please ensure the FastAPI server is running.")
                st.error(f"Error details: {e}")

