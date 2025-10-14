# 🛍️ Visual Product Matcher

Approach documentation- https://drive.google.com/file/d/1vkspviINRi7-ySTC-AdEGrfHsqq-MUIu/view?usp=sharing

AI-powered fashion product similarity search using Vision Transformer and FAISS.

## ✨ Features

- 📸 Upload images or use URLs to find similar products
- 🎯 Filter results by similarity score (0-100%)
- 📊 View detailed product metadata
- 🚀 Fast FAISS-powered search across 250 fashion products
- 🎨 Modern, responsive UI with dark theme

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- 4GB RAM minimum
- Internet connection

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/bishal692002/Visual-Product-Matcher.git
cd Visual-Product-Matcher
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 🎯 Usage

1. Choose input method (Upload Image or Image URL)
2. Select or paste your fashion product image
3. Adjust filters:
   - **Minimum Similarity**: Set threshold (0-100%)
   - **Results to Show**: Choose number of results (3-10)
4. Click "Find Similar Products"
5. Browse results with similarity scores and product details

## 🏗️ Tech Stack

- **Frontend**: Streamlit with custom CSS
- **ML Model**: Vision Transformer (google/vit-base-patch16-224)
- **Vector Search**: FAISS
- **Dataset**: HuggingFace Hub (250 Myntra products)
- **Image Processing**: PIL, transformers

## 📁 Project Structure

```
Visual-Product-Matcher/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md          # Documentation
└── img/
    ├── styles.csv     # Product metadata
    └── images/        # Product images
```

## 🌐 Deployment

### Streamlit Cloud

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Select repository: `bishal692002/Visual-Product-Matcher`
4. Set main file: `app.py`
5. Click Deploy

## 📊 Dataset

- **Source**: Myntra Fashion Dataset
- **Size**: 250 curated products
- **Storage**: HuggingFace Hub (`Gauravannad/fashion-products-embeddings`)
- **Categories**: Apparel, Footwear, Accessories
- **Metadata**: Name, category, color, gender, season, usage

## � Configuration

Change the dataset in `app.py` (line 235):
```python
DATASET_REPO = "your-username/your-dataset-embeddings"
```

## 🐛 Troubleshooting

**Port already in use:**
```bash
streamlit run app.py --server.port 8502
```

**Dataset not loading:**
- Check internet connection
- Verify HuggingFace dataset is public

**Memory errors:**
- Close other applications
- Reduce `num_results` slider value

## 👨‍💻 Developer

**Bishal Biswas**
- GitHub: [@bishal692002](https://github.com/bishal692002)

## 📝 License

MIT License - feel free to use this project for learning and development.

## � Acknowledgments

- Google for Vision Transformer model
- HuggingFace for model hosting
- Myntra for fashion dataset
- Streamlit community

---

**Built with ❤️ using AI and Open Source**
