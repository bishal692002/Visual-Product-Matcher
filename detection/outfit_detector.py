"""
OutfitDetector - YOLOv8-based person outfit item detection.

Detects clothing items (tops, bottoms, shoes) from outfit images using YOLOv8.
Uses position-based heuristics to categorize detected items by vertical location.
"""

import torch
from PIL import Image
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO


class OutfitDetector:
    """
    YOLOv8-based detector for outfit item detection and categorization.
    
    Attributes:
        model: Loaded YOLOv8 model
        device: Compute device (mps/cuda/cpu)
        model_size: Model variant ("n", "s", "m", "l", "x")
    """
    
    def __init__(self, model_size: str = "s", device: str = None):
        """
        Initialize YOLOv8 detector.
        
        Args:
            model_size: Model size ("n"ano, "s"mall, "m"edium, "l"arge, "x"large)
            device: Device to use ("mps", "cuda", "cpu"). Auto-detect if None.
        """
        print(f"Loading YOLOv8{model_size} model...")
        
        # Auto-detect device if not specified
        if device is None:
            self.device = self._get_device()
        else:
            self.device = device
        
        self.model_size = model_size
        self.model = YOLO(f"yolov8{model_size}.pt")
        self.class_names = self.model.names if hasattr(self.model, "names") else {}
        
        print(f"✓ YOLOv8{model_size} loaded on {self.device}")
    
    def detect_items(self, image: Image.Image, conf_threshold: float = 0.25) -> List[Dict]:
        """
        Detect clothing items in image and categorize by position.
        
        Args:
            image: PIL Image of outfit
            conf_threshold: Confidence threshold for detections (0.0-1.0)
        
        Returns:
            List of detection dicts:
            [
                {
                    "item_id": int,
                    "category": str ("top", "bottom", "shoes", "accessories"),
                    "confidence": float (0.0-1.0),
                    "bbox": tuple (x_min, y_min, x_max, y_max),
                    "cropped_image": PIL.Image
                },
                ...
            ]
        """
        # Run YOLOv8 inference
        results = self.model(image, conf=conf_threshold, verbose=False)
        detections: List[Dict] = []
        width, height = image.size
        image_shape = (height, width, 3)

        raw_boxes: List[Dict] = []
        for result in results:
            boxes = result.boxes

            for box in boxes:
                # Get bounding box in (x_min, y_min, x_max, y_max) format
                xyxy = box.xyxy[0].cpu().numpy()
                x_min, y_min, x_max, y_max = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

                # Get confidence
                confidence = float(box.conf[0].cpu().numpy())

                cls_id = int(box.cls[0].cpu().numpy()) if box.cls is not None else -1
                class_name = str(self.class_names.get(cls_id, "unknown")).lower()
                raw_boxes.append(
                    {
                        "bbox": (x_min, y_min, x_max, y_max),
                        "confidence": confidence,
                        "class_id": cls_id,
                        "class_name": class_name,
                    }
                )

        # YOLO COCO does not directly classify tops/bottoms/shoes reliably.
        # We anchor on the most prominent person and split into garment zones.
        person_boxes = [b for b in raw_boxes if b["class_name"] == "person"]
        primary_person = self._select_primary_person(person_boxes, width, height)

        if primary_person:
            zone_detections = self._create_person_zone_detections(primary_person, image, image_shape)
            detections.extend(zone_detections)
        else:
            # Fallback: classify all detections by vertical position and keep best per category.
            by_category: Dict[str, Dict] = {}
            for b in raw_boxes:
                category = self._classify_item_by_position(b["bbox"], image_shape)
                previous = by_category.get(category)
                if previous is None or b["confidence"] > previous["confidence"]:
                    by_category[category] = {
                        "category": category,
                        "confidence": b["confidence"],
                        "bbox": b["bbox"],
                    }

            for category in ("top", "bottom", "shoes"):
                selected = by_category.get(category)
                if selected is None:
                    continue
                x_min, y_min, x_max, y_max = selected["bbox"]
                detections.append(
                    {
                        "item_id": len(detections),
                        "category": category,
                        "confidence": float(selected["confidence"]),
                        "bbox": (x_min, y_min, x_max, y_max),
                        "cropped_image": image.crop((x_min, y_min, x_max, y_max)),
                    }
                )

        # Ensure deterministic category order for downstream grouped rendering.
        order = {"top": 0, "bottom": 1, "shoes": 2}
        detections.sort(key=lambda d: order.get(d.get("category", ""), 99))
        for idx, det in enumerate(detections):
            det["item_id"] = idx
        
        return detections

    def _select_primary_person(
        self,
        person_boxes: List[Dict],
        image_width: int,
        image_height: int,
    ) -> Optional[Dict]:
        """
        Choose the best person box by balancing area and confidence.
        """
        if not person_boxes:
            return None

        image_area = max(1, image_width * image_height)

        def score(box: Dict) -> float:
            x_min, y_min, x_max, y_max = box["bbox"]
            box_area = max(1, (x_max - x_min) * (y_max - y_min))
            area_ratio = box_area / image_area
            return (0.7 * area_ratio) + (0.3 * float(box["confidence"]))

        return max(person_boxes, key=score)

    def _create_person_zone_detections(
        self,
        person_box: Dict,
        image: Image.Image,
        image_shape: Tuple[int, int, int],
    ) -> List[Dict]:
        """
        Split the detected person into TOP/BOTTOM/SHOES zones.
        """
        height, width = image_shape[0], image_shape[1]
        x_min, y_min, x_max, y_max = person_box["bbox"]
        conf = float(person_box["confidence"])

        # Expand slightly to avoid cutting garment boundaries too tightly.
        pad_x = int(0.04 * max(1, x_max - x_min))
        pad_y = int(0.03 * max(1, y_max - y_min))
        x_min = max(0, x_min - pad_x)
        x_max = min(width, x_max + pad_x)
        y_min = max(0, y_min - pad_y)
        y_max = min(height, y_max + pad_y)

        person_h = max(1, y_max - y_min)

        # Tuned splits for full-body fashion photos:
        # - keep top through ~50%
        # - keep bottom through ~88%
        # - reserve final ~12% primarily for footwear so jeans hems don't dominate shoe crop
        split_top = y_min + int(person_h * 0.50)
        split_bottom = y_min + int(person_h * 0.88)

        # Guarantee minimum zone heights so small person boxes remain usable.
        min_zone_h = 20
        if (split_top - y_min) < min_zone_h:
            split_top = min(y_max - 2 * min_zone_h, y_min + min_zone_h)
        if (split_bottom - split_top) < min_zone_h:
            split_bottom = min(y_max - min_zone_h, split_top + min_zone_h)
        if (y_max - split_bottom) < min_zone_h:
            split_bottom = max(y_min + 2 * min_zone_h, y_max - min_zone_h)

        zones = [
            ("top", (x_min, y_min, x_max, split_top), max(0.35, conf)),
            ("bottom", (x_min, split_top, x_max, split_bottom), max(0.30, conf * 0.95)),
            ("shoes", (x_min, split_bottom, x_max, y_max), max(0.25, conf * 0.90)),
        ]

        detections: List[Dict] = []
        for category, bbox, zone_conf in zones:
            zx_min, zy_min, zx_max, zy_max = bbox
            if (zx_max - zx_min) < 16 or (zy_max - zy_min) < 16:
                continue
            detections.append(
                {
                    "item_id": len(detections),
                    "category": category,
                    "confidence": float(zone_conf),
                    "bbox": (zx_min, zy_min, zx_max, zy_max),
                    "cropped_image": image.crop((zx_min, zy_min, zx_max, zy_max)),
                }
            )

        return detections
    
    def _classify_item_by_position(self, bbox: Tuple[int, int, int, int], 
                                   image_shape: Tuple[int, int, int]) -> str:
        """
        Categorize detected item by vertical position in image.
        
        Uses simple heuristic:
        - Top 30% → "top"
        - Middle 40% (30-70%) → "bottom"
        - Bottom 30% (70-100%) → "shoes"
        
        Args:
            bbox: Bounding box (x_min, y_min, x_max, y_max)
            image_shape: Image shape (height, width, channels)
        
        Returns:
            Category string
        """
        x_min, y_min, x_max, y_max = bbox
        height = image_shape[0]
        
        # Calculate vertical center of bounding box as percentage
        bbox_center_y = (y_min + y_max) / 2
        y_percentage = bbox_center_y / height

        bbox_height = max(1, y_max - y_min)
        bbox_height_ratio = bbox_height / max(1, height)
        y_min_ratio = y_min / max(1, height)
        y_max_ratio = y_max / max(1, height)

        # Shoes tend to appear at the very bottom and are often relatively short boxes.
        if y_min_ratio >= 0.82:
            return "shoes"
        if y_max_ratio >= 0.92 and bbox_height_ratio <= 0.24:
            return "shoes"
        
        # Classify based on position
        if y_percentage < 0.38:
            return "top"
        elif y_percentage < 0.80:
            return "bottom"
        else:
            return "shoes"
    
    def _get_device(self) -> str:
        """
        Auto-detect compute device.
        
        Priority: MPS (Apple) > CUDA (Nvidia) > CPU
        
        Returns:
            Device string ("mps", "cuda", or "cpu")
        """
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
