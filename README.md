# Visual Product Matcher# 🛍️ Visual Product Matcher



AI-powered fashion product similarity search using image recognition.A web application that helps users find visually similar fashion products based on uploaded images using deep learning and image embeddings.



## Features## 🌐 Live Demo

- 📸 Upload images or use URLs to find similar products

- 🎯 Filter results by similarity score*[Your deployed URL will go here]*

- 📊 View product metadata and details

- 🚀 Fast FAISS-powered search across 250+ products## ✨ Features



## Quick Start- **Image Upload**: Upload product images directly from your device

- **Image URL Input**: Provide a URL to fetch and analyze images from the web

1. **Install dependencies:**- **Visual Similarity Search**: Find products that look similar to your input image

```bash- **Smart Filtering**: Filter results by similarity score threshold

pip install -r requirements.txt- **Product Metadata**: View detailed product information (category, color, gender, etc.)

```- **Adjustable Results**: Choose how many recommendations to display (3-10)

- **Mobile Responsive**: Works seamlessly on desktop and mobile devices

2. **Start Backend (Terminal 1):**- **Fast AI-Powered Search**: Uses Vision Transformer (ViT) for image embeddings

```bash- **FAISS indexing**: Lightning-fast similarity search across thousands of products

export KMP_DUPLICATE_LIB_OK=TRUE

source venv/bin/activate## 🏗️ Architecture

python -m uvicorn main:app --reload

```### Overview



3. **Start Frontend (Terminal 2):**The application consists of three main components:

```bash

source venv/bin/activate1. **Frontend (Streamlit)**: User interface for uploading images and displaying results

streamlit run app.py2. **Backend (FastAPI)**: API server that processes images and performs similarity search

```3. **Dataset (Hugging Face)**: Pre-computed image embeddings stored on Hugging Face Hub



4. **Open:** http://localhost:8502### Technical Stack



## Tech Stack- **Frontend**: Streamlit

- FastAPI + Vision Transformer (ViT)- **Backend**: FastAPI + Uvicorn

- Streamlit UI- **ML Model**: Google's Vision Transformer (ViT-base-patch16-224)

- FAISS similarity search- **Vector Search**: FAISS (Facebook AI Similarity Search)

- HuggingFace datasets- **Image Processing**: PIL (Pillow)

- **Dataset Storage**: Hugging Face Hub

## Dataset- **Product Data**: Myntra Fashion Dataset (44K+ products)

250 curated Myntra fashion products hosted on HuggingFace: `Gauravannad/fashion-products-embeddings`

### How It Works

1. User uploads an image or provides a URL
2. Frontend sends the image to the FastAPI backend
3. Backend extracts image embeddings using ViT model
4. FAISS index finds the most similar product embeddings
5. Backend retrieves product metadata and returns results
6. Frontend displays similar products with similarity scores

## 📋 Requirements

- Python 3.8+
- Internet connection for downloading models and datasets
- Hugging Face account (for dataset hosting)
- At least 4GB RAM
- 2GB free disk space

## 🚀 Setup Instructions

### Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd visual-product-matcher
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Prepare Dataset

#### Option A: Use Existing Dataset
If you already have a dataset on Hugging Face:

1. Update `main.py` line 33:
```python
DATASET_REPO = "your-username/your-dataset-embeddings"
```

#### Option B: Create New Dataset

1. Download the [Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) from Kaggle

2. Extract and copy images:
   - Place all `.jpg` files from `images/` folder into `./img/images/`
   - Copy `styles.csv` to `./img/styles.csv`

3. Get your Hugging Face token:
   - Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Create a new token with **write** permissions
   - Copy the token

4. Update `upload_ds.py`:
```python
repo_name = 'your-username/fashion-products'
hf_token = 'your_token_here'
```

5. Run the dataset upload script:
```bash
python upload_ds.py
```
*Note: This will take 30-60 minutes depending on your internet speed*

6. Update `main.py` with your dataset name:
```python
DATASET_REPO = "your-username/fashion-products-embeddings"
```

### Step 5: Run the Application

#### Terminal 1 - Start Backend Server

```bash
# Activate virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Start FastAPI server
uvicorn main:app --reload
```

Wait until you see:
```
INFO:     Application startup complete.
```

#### Terminal 2 - Start Frontend

```bash
# Activate virtual environment in new terminal
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Start Streamlit app
streamlit run app.py
```

The app should automatically open in your browser at `http://localhost:8501`

## 🎯 Usage

1. **Choose Input Method**:
   - Select "Upload Image" to upload from your device
   - Select "Image URL" to provide an image link

2. **Upload/Enter Image**:
   - For upload: Click "Browse files" and select an image
   - For URL: Paste the image URL

3. **Adjust Filters** (Optional):
   - Set minimum similarity score (0-100%)
   - Choose number of results to display (3-10)

4. **Find Similar Products**:
   - Click the "Find Similar Products" button
   - Wait for AI to process your image

5. **View Results**:
   - Browse through recommended similar products
   - Check similarity scores
   - Expand "Product Details" to see more information

## 📁 Project Structure

```
visual-product-matcher/
├── app.py                 # Streamlit frontend application
├── main.py               # FastAPI backend server
├── Embed.py              # Embedding generation utilities
├── upload_ds.py          # Script to upload dataset to HuggingFace
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── img/                 # Image dataset folder
    ├── images/          # Product images (*.jpg)
    └── styles.csv       # Product metadata
```

## 🌐 Deployment

### Option 1: Streamlit Cloud (Recommended for Frontend)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy `app.py`
4. **Note**: You'll need to deploy the backend separately

### Option 2: Hugging Face Spaces

1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces)
2. Choose "Streamlit" as SDK
3. Upload files or connect to GitHub repository
4. Add secrets for HuggingFace token if needed

### Option 3: Railway/Render (For Full Stack)

1. Deploy backend (FastAPI) on Railway/Render
2. Deploy frontend (Streamlit) on Streamlit Cloud
3. Update `BACKEND_URL` in `app.py` with your deployed backend URL

### Option 4: Docker (Advanced)

Create `Dockerfile`:

```dockerfile
FROM python:3.9

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000 8501

CMD uvicorn main:app --host 0.0.0.0 --port 8000 & streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Build and run:
```bash
docker build -t visual-matcher .
docker run -p 8000:8000 -p 8501:8501 visual-matcher
```

## 🔧 Configuration

### Adjust Number of Recommendations

In `main.py`, line 85:
```python
scores, retrieved_examples = dataset.get_nearest_examples(
    "embeddings", query_embedding, k=10  # Change this number
)
```

### Change AI Model

In `main.py`, line 27:
```python
MODEL_CKPT = 'google/vit-base-patch16-224'  # Try other models
```

### Customize UI

Edit `app.py` to modify:
- Page title and icon (line 9-12)
- Column layout
- Color schemes
- Filter options

## 📊 Dataset Information

- **Source**: Myntra Fashion Product Images
- **Size**: 44,000+ products
- **Categories**: Men, Women, Boys, Girls
- **Product Types**: Apparel, Footwear, Accessories
- **Metadata**: Product name, category, color, gender, season, usage

## 🐛 Troubleshooting

### Backend Not Starting

```bash
# Check if port 8000 is already in use
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# Use different port
uvicorn main:app --port 8001
```

### Dataset Not Loading

- Check internet connection
- Verify HuggingFace token has read permissions
- Ensure dataset repository is public or token has access

### Memory Errors

- Reduce batch size in model loading
- Close other applications
- Consider using smaller model variant

### CORS Errors

- Ensure CORS middleware is enabled in `main.py`
- Check frontend URL matches allowed origins

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

This project is licensed under the MIT License.

## 👏 Acknowledgments

- Google for the Vision Transformer model
- Hugging Face for model hosting and datasets
- Myntra for the fashion product dataset
- Streamlit and FastAPI communities

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Built with ❤️ using AI and Open Source**
