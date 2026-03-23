"""
Visualization utilities for outfit detection results.

Provides functions to draw bounding boxes and create summary reports.
"""

import cv2
import numpy as np
from PIL import Image
from typing import List, Dict


def draw_detections_on_image(image: Image.Image, detections: List[Dict]) -> Image.Image:
    """
    Draw bounding boxes and labels on image.
    
    Args:
        image: PIL Image
        detections: List of detection dicts with keys:
                   - "bbox": (x_min, y_min, x_max, y_max)
                   - "category": str
                   - "confidence": float
    
    Returns:
        PIL Image with drawn detections
    """
    # Convert PIL to numpy array for OpenCV
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Color map for categories
    colors = {
        "top": (0, 255, 0),        # Green
        "bottom": (255, 0, 0),     # Blue
        "shoes": (0, 0, 255),      # Red
        "accessories": (255, 255, 0)  # Cyan
    }
    
    for det in detections:
        x_min, y_min, x_max, y_max = det["bbox"]
        category = det["category"]
        confidence = det["confidence"]
        
        # Get color
        color = colors.get(category, (128, 128, 128))  # Gray default
        
        # Draw rectangle
        cv2.rectangle(img_cv, (x_min, y_min), (x_max, y_max), color, 2)
        
        # Draw label
        label = f"{category.capitalize()} {confidence:.0%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        
        # Get text size for background
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        text_x = x_min
        text_y = y_min - 5
        
        # Draw background rectangle for text
        cv2.rectangle(img_cv, 
                     (text_x, text_y - text_size[1] - 5),
                     (text_x + text_size[0], text_y + 5),
                     color, -1)
        
        # Draw text
        cv2.putText(img_cv, label, (text_x, text_y), font, 
                   font_scale, (255, 255, 255), thickness)
    
    # Convert back to PIL
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def create_detection_summary(detections: List[Dict]) -> str:
    """
    Create a text summary of detections.
    
    Args:
        detections: List of detection dicts
    
    Returns:
        Formatted string summary
    """
    if not detections:
        return "No items detected."
    
    lines = []
    lines.append(f"Detected {len(detections)} item(s):")
    lines.append("")
    
    # Group by category
    by_category = {}
    for det in detections:
        cat = det["category"].capitalize()
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(det)
    
    # Format each category
    for category in sorted(by_category.keys()):
        items = by_category[category]
        lines.append(f"• {category} ({len(items)} item{'s' if len(items) > 1 else ''})")
        for item in items:
            conf_pct = f"{item['confidence']:.0%}"
            lines.append(f"  - ID {item['item_id']}: {conf_pct} confidence")
    
    return "\n".join(lines)


def generate_summary(detections: List[Dict]) -> str:
    """Alias for create_detection_summary for backward compatibility."""
    return create_detection_summary(detections)
