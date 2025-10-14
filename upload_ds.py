from datasets import load_dataset, Dataset, Features, Image as HFImage, Value
import os
import csv
from Embed import create_dataset_embeddings
from PIL import Image

# Image folder
img_folder = './img/images'  # Point directly to images folder
styles_csv = './img/styles.csv'
repo_name = 'Gauravannad/fashion-products'
hf_token = os.getenv('HUGGINGFACE_TOKEN', 'your-token-here')  # Use environment variable

print("📊 Creating dataset from images and metadata...")

# Load metadata
import pandas as pd
df = pd.read_csv(styles_csv, on_bad_lines='skip')

# Get list of image files
image_files = sorted([f for f in os.listdir(img_folder) if f.endswith('.jpg')])
print(f"Found {len(image_files)} images")

# Create dataset dict
data_dict = {
    'image': [],
    'file_name': [],
    'id': []
}

for img_file in image_files:
    img_path = os.path.join(img_folder, img_file)
    img_id = int(img_file.replace('.jpg', ''))
    
    data_dict['image'].append(img_path)
    data_dict['file_name'].append(img_file)
    data_dict['id'].append(img_id)

# Create dataset
features = Features({
    'image': HFImage(),
    'file_name': Value('string'),
    'id': Value('int64')
})

dataset = Dataset.from_dict(data_dict, features=features)
print(f"✅ Created dataset with {len(dataset)} examples")

# Push to hub
print(f"📤 Uploading to {repo_name}...")
dataset.push_to_hub(repo_name, token=hf_token)
print("✅ Dataset uploaded!")

# Create embeddings dataset
print("🧠 Creating embeddings...")
create_dataset_embeddings(input_dataset=repo_name, output_dataset=repo_name + '-embeddings', token=hf_token)

print('✅ Done!')



