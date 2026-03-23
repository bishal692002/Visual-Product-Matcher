"""
OutfitSearchService - Integrates detection, embedding, and FAISS search.

Pipeline:
    Input Image → Detection (YOLOv8)
               → For each item:
                   - Extract embedding (FashionCLIP)
                   - FAISS similarity search
                   - Retrieve metadata
               → Output grouped results per category
"""

import time
import numpy as np
import torch
from PIL import Image
from typing import List, Dict, Tuple
from transformers import CLIPProcessor, CLIPModel

from detection.outfit_detector import OutfitDetector
from detection.viz_utils import draw_detections_on_image
from .recommendation import OutfitRecommender


class OutfitSearchService:
    """
    Full outfit analysis service combining detection, embedding, and search.
    
    Attributes:
        detector: YOLOv8 outfit detector
        processor: CLIP processor for image preprocessing
        model: FashionCLIP model for embedding extraction
        dataset: HuggingFace dataset with FAISS index
        metadata_df: DataFrame with product metadata (optional, for enrichment)
        device: Compute device (mps/cuda/cpu)
    """
    
    def __init__(self, processor: CLIPProcessor, model: CLIPModel, 
                 dataset, metadata_df=None):
        """
        Initialize the outfit search service.
        
        Args:
            processor: CLIP processor (from transformers)
            model: FashionCLIP model (from transformers)
            dataset: HuggingFace dataset with embeddings and FAISS index
            metadata_df: Optional pandas DataFrame with product metadata
        """
        print("Initializing OutfitSearchService...")
        
        self.processor = processor
        self.model = model
        self.dataset = dataset
        self.metadata_df = metadata_df
        self.device = self._get_device()
        
        # Initialize detector (lightweight, loaded once)
        self.detector = OutfitDetector(model_size="s")
        
        # Initialize recommender for outfit suggestions
        self.recommender = OutfitRecommender(dataset, metadata_df)
        
        print("OutfitSearchService ready.")
    
    def analyze_outfit(self, image: Image.Image, 
                      top_k_per_item: int = 10) -> Dict:
        """
        Analyze outfit image: detect items, generate embeddings, search for matches.
        
        This is the main entry point for full outfit analysis.
        
        Args:
            image: PIL Image of outfit (any size)
            top_k_per_item: Number of similar products to retrieve per item
        
        Returns:
            Dict with structure:
            {
                "detected_items": [
                    {
                        "item_id": 0,
                        "category": "top",
                        "confidence": 0.92,
                        "cropped_image": PIL.Image,
                        "search_results": [
                            {
                                "similarity_score": 87,  # 0-100%
                                "image": PIL.Image,
                                "metadata": {
                                    "productDisplayName": "...",
                                    "category": "...",
                                    ...
                                }
                            },
                            ...
                        ]
                    },
                    ...
                ],
                "outfit_image_with_boxes": PIL.Image,  # Visualization
                "outfit_recommendations": {...},  # From recommendation engine
                "analysis_time_ms": float,  # Total time taken
                "error": str (optional, if something went wrong)
            }
        """
        import time
        start_time = time.time()
        
        # 1. DETECT items in outfit
        print(f"Detecting items...")
        detections = self.detector.detect_items(image, conf_threshold=0.5)
        
        if not detections:
            return {
                "error": "No items detected. Please upload a clear outfit image.",
                "detected_items": [],
                "outfit_image_with_boxes": None,
                "outfit_recommendations": {},
                "analysis_time_ms": (time.time() - start_time) * 1000
            }
        
        print(f"Detected {len(detections)} items")
        
        # 2. For each item: extract embedding → FAISS search
        outfit_analysis = {
            "detected_items": [],
            "outfit_image_with_boxes": None,
            "outfit_recommendations": {},
            "analysis_time_ms": 0,
            "error": None
        }
        
        for detection in detections:
            cropped_image = detection["cropped_image"]
            item_id = detection["item_id"]
            category = detection["category"]
            confidence = detection["confidence"]
            
            print(f"  Processing item {item_id}: {category} (conf: {confidence:.1%})")
            
            # 2a. Extract embedding for cropped item
            try:
                embedding = self._extract_embedding(cropped_image)  # 512-dim
            except Exception as e:
                print(f"    ✗ Embedding failed: {e}")
                embedding = np.zeros(512)  # Fallback
            
            # 2b. FAISS search (top-K per item)
            try:
                scores, results = self.dataset.get_nearest_examples(
                    "embeddings_norm",
                    embedding,
                    k=min(max(top_k_per_item * 6, top_k_per_item), len(self.dataset))
                )
                print(f"    ✓ Found {len(results['image'])} matches")
            except Exception as e:
                print(f"    ✗ Search failed: {e}")
                scores, results = [], {}
            
            # 2c. Format results with metadata
            search_results = self._format_search_results(scores, results, expected_category=category)

            top_meta = search_results[0]["metadata"] if search_results else {}
            
            outfit_analysis["detected_items"].append({
                "item_id": item_id,
                "category": category,
                "confidence": confidence,
                "cropped_image": cropped_image,
                "search_results": search_results,
                "gender": top_meta.get("gender", "Unknown"),
                "color": top_meta.get("baseColour", "Unknown"),
                "articleType": top_meta.get("articleType", "Unknown"),
                "productDisplayName": top_meta.get("productDisplayName", "Unknown"),
            })
        
        # 3. Draw detections on original image for visualization
        outfit_analysis["outfit_image_with_boxes"] = draw_detections_on_image(
            image, detections
        )
        
        # 4. Generate outfit recommendations based on detected items
        outfit_analysis["outfit_recommendations"] = self.recommender.recommend_outfit_complements(
            detected_items=outfit_analysis["detected_items"],
            k=5
        )
        
        outfit_analysis["analysis_time_ms"] = (time.time() - start_time) * 1000
        print(f"Analysis complete in {outfit_analysis['analysis_time_ms']:.0f}ms")
        
        return outfit_analysis
    
    def _extract_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Extract L2-normalized FashionCLIP embedding for an image.
        
        Uses the raw 768-dim ViT pooler output (NOT the 512-dim visual_projection).
        This matches the embeddings stored in the FAISS index.
        
        Why 768-dim?
        - Richer visual features (color, texture, garment shape)
        - Matches the training embeddings in the dataset
        - Provides better quality for image-to-image search
        
        Args:
            image: PIL Image
        
        Returns:
            768-dim L2-normalized numpy array
        """
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        
        with torch.no_grad():
            vision_out = self.model.vision_model(pixel_values=pixel_values)
            features = vision_out.pooler_output  # 768-dim (NOT visual_projection)
        
        vec = features.squeeze().cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec
    
    def _format_search_results(self, scores: np.ndarray,
                              retrieved: Dict,
                              expected_category: str = "") -> List[Dict]:
        """
        Format FAISS search results with metadata for display.
        
        Args:
            scores: Array of cosine similarity scores
            retrieved: Dict with dataset columns (image, metadata, etc.)
        
        Returns:
            List of formatted result dicts
        """
        results = []
        
        expected_category = (expected_category or "").lower()
        images = retrieved.get("image", [])
        for i in range(len(images)):
            try:
                # Cosine similarity score (0-1 range) → 0-100% for display
                similarity_pct = round(max(0, min(1.0, float(scores[i]))) * 100, 1)
                
                # Extract metadata fields
                def _get_field(col, default="Unknown"):
                    col_data = retrieved.get(col)
                    if col_data is None or i >= len(col_data):
                        return default
                    val = col_data[i]
                    return str(val) if val not in (None, "", "nan") else default
                
                result = {
                    "similarity_score": similarity_pct,
                    "image": images[i],
                    "metadata": {
                        "productDisplayName": _get_field("productDisplayName"),
                        "masterCategory": _get_field("masterCategory"),
                        "subCategory": _get_field("subCategory"),
                        "articleType": _get_field("articleType"),
                        "baseColour": _get_field("baseColour"),
                        "gender": _get_field("gender"),
                        "season": _get_field("season"),
                        "usage": _get_field("usage"),
                    }
                }

                normalized_type = self._normalize_item_type(
                    result["metadata"].get("articleType", ""),
                    result["metadata"].get("subCategory", ""),
                    result["metadata"].get("productDisplayName", ""),
                )

                if expected_category in {"top", "bottom", "shoes"} and normalized_type != expected_category:
                    continue

                results.append(result)
            
            except Exception as e:
                print(f"  ⚠ Error formatting result {i}: {e}")
                continue
        
        # If strict filtering was too aggressive, gracefully fall back to original rank order.
        if not results:
            images = retrieved.get("image", [])
            for i in range(len(images)):
                try:
                    similarity_pct = round(max(0, min(1.0, float(scores[i]))) * 100, 1)

                    def _get_field(col, default="Unknown"):
                        col_data = retrieved.get(col)
                        if col_data is None or i >= len(col_data):
                            return default
                        val = col_data[i]
                        return str(val) if val not in (None, "", "nan") else default

                    results.append(
                        {
                            "similarity_score": similarity_pct,
                            "image": images[i],
                            "metadata": {
                                "productDisplayName": _get_field("productDisplayName"),
                                "masterCategory": _get_field("masterCategory"),
                                "subCategory": _get_field("subCategory"),
                                "articleType": _get_field("articleType"),
                                "baseColour": _get_field("baseColour"),
                                "gender": _get_field("gender"),
                                "season": _get_field("season"),
                                "usage": _get_field("usage"),
                            },
                        }
                    )
                except Exception:
                    continue

        return results

    def _normalize_item_type(self, article_type: str, sub_category: str, product_name: str) -> str:
        """
        Map product text metadata into one of: top, bottom, shoes, accessories.
        """
        text = " ".join(
            [
                str(article_type or "").lower(),
                str(sub_category or "").lower(),
                str(product_name or "").lower(),
            ]
        )

        shoes_kw = ["shoe", "sneaker", "heel", "boot", "loafer", "sandal", "flip flop", "slipper"]
        bottom_kw = ["jean", "trouser", "pant", "short", "skirt", "bottomwear", "capri", "legging", "track pant"]
        top_kw = ["shirt", "t-shirt", "tshirt", "top", "blouse", "hoodie", "sweater", "kurta", "jacket", "polo", "tee"]

        if any(k in text for k in shoes_kw):
            return "shoes"
        if any(k in text for k in bottom_kw):
            return "bottom"
        if any(k in text for k in top_kw):
            return "top"
        return "accessories"
    
    def _get_device(self) -> str:
        """
        Detect best available device for inference.
        Priority: MPS > CUDA > CPU
        """
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
