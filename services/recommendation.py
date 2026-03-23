"""
OutfitRecommender - Suggest complementary items based on detected items.

Strategy (Phase 1):
    Use metadata rules to find complementary items:
    - For a detected "top" → recommend "bottoms" + "shoes"
    - For a detected "bottom" → recommend "tops" + "shoes"  
    - For a detected "shoes" → recommend "tops" + "bottoms"
    
    Matching rules:
    - Same gender (if available)
    - Optional: similar color palette
    - Sample top matches from available inventory
    
Phase 2 Enhancement:
    Replace with embedding-based compatibility model trained on outfit data.
"""

import re
from collections import Counter
from typing import List, Dict, Tuple

import pandas as pd


class OutfitRecommender:
    """
    Recommends complementary clothing items based on detected outfit items.
    
    Currently uses metadata-based filtering (fast, interpretable).
    Can be upgraded to embedding-based in Phase 2.
    
    Attributes:
        dataset: HuggingFace dataset with product embeddings
        metadata_df: pandas DataFrame with product metadata
    """
    
    # Mapping of item categories to complementary categories
    COMPLEMENTS = {
        "top": ["bottom", "shoes"],
        "bottom": ["top", "shoes"],
        "shoes": ["top", "bottom"],
        "accessories": ["top", "bottom", "shoes"]
    }
    
    def __init__(self, dataset, metadata_df=None):
        """
        Initialize recommendation engine.
        
        Args:
            dataset: HuggingFace dataset (used for item count/validation)
            metadata_df: pandas DataFrame with product metadata
                        Should have columns: masterCategory, gender, baseColour, etc.
        """
        self.dataset = dataset
        self.metadata_df = metadata_df

        if self.metadata_df is not None and not self.metadata_df.empty:
            self.metadata_df = self.metadata_df.copy()
            for col in ["productDisplayName", "articleType", "subCategory", "baseColour", "gender"]:
                if col not in self.metadata_df.columns:
                    self.metadata_df[col] = ""

            self.metadata_df["normalized_type"] = self.metadata_df.apply(
                lambda r: self._normalize_item_type(
                    r.get("articleType", ""),
                    r.get("subCategory", ""),
                    r.get("productDisplayName", ""),
                ),
                axis=1,
            )
            self.metadata_df["color_norm"] = self.metadata_df["baseColour"].astype(str).str.lower().str.strip()
            self.metadata_df["gender_norm"] = self.metadata_df["gender"].astype(str).str.lower().str.strip()
    
    def recommend_outfit_complements(self, detected_items: List[Dict],
                                     k: int = 5) -> Dict:
        """
        Generate outfit recommendations based on detected items.
        
        Args:
            detected_items: List of detected items from OutfitSearchService
                           Each item has: item_id, category, confidence, etc.
            k: Number of recommendations per complementary category
        
        Returns:
            Dict with structure:
            {
                "complementary_items": [
                    {
                        "detected_item_id": 0,
                        "detected_category": "top",
                        "recommendations": [
                            {
                                "category": "bottom",
                                "items": [
                                    {
                                        "product_id": "...",
                                        "product_name": "...",
                                        "category": "...",
                                        "color": "...",
                                        "gender": "...",
                                        "reason": "Complements your top..."
                                    },
                                    ...
                                ]
                            },
                            ...
                        ]
                    },
                    ...
                ]
            }
        """
        recommendations = {"complementary_items": []}
        
        if self.metadata_df is None or self.metadata_df.empty:
            print("⚠ No metadata available for recommendations")
            return recommendations
        
        detected_categories = {
            str(item.get("category", "")).lower()
            for item in detected_items
            if str(item.get("category", "")).lower() in {"top", "bottom", "shoes"}
        }

        target_categories = self._target_categories(detected_categories)
        context = self._build_outfit_context(detected_items)

        outfit_level = {
            "detected_item_id": -1,
            "detected_category": "outfit",
            "recommendations": [],
        }

        for complement_cat in target_categories:
            matches = self._find_matches_by_category(
                context=context,
                complement_category=complement_cat,
                k=k,
            )
            outfit_level["recommendations"].append({
                "category": complement_cat,
                "items": matches,
            })

        recommendations["complementary_items"].append(outfit_level)
        
        return recommendations

    def _target_categories(self, detected_categories: set) -> List[str]:
        """
        Decide which categories to recommend based on what is already detected.
        """
        full_set = {"top", "bottom", "shoes"}
        missing = [cat for cat in ("top", "bottom", "shoes") if cat not in detected_categories]

        # Prioritize missing pieces, but still suggest alternates from all categories
        # when a full outfit is already present.
        if missing:
            return missing
        return [cat for cat in ("top", "bottom", "shoes") if cat in full_set]

    def _build_outfit_context(self, detected_items: List[Dict]) -> Dict:
        """
        Build aggregated outfit context from all detected items.
        """
        colors = []
        genders = []
        text_corpus = []
        categories = []

        for item in detected_items:
            cat = str(item.get("category", "")).lower().strip()
            if cat:
                categories.append(cat)

            col = str(item.get("color", "")).lower().strip()
            if col and col not in {"", "unknown", "n/a", "nan"}:
                colors.append(col)

            gender = str(item.get("gender", "")).lower().strip()
            if gender and gender not in {"", "unknown", "n/a", "nan"}:
                genders.append(gender)

            name = str(item.get("productDisplayName", ""))
            article = str(item.get("articleType", ""))
            text_corpus.append(f"{name} {article}")

            for r in item.get("search_results", [])[:3]:
                meta = r.get("metadata", {})
                mcolor = str(meta.get("baseColour", "")).lower().strip()
                if mcolor and mcolor not in {"", "unknown", "n/a", "nan"}:
                    colors.append(mcolor)
                mgender = str(meta.get("gender", "")).lower().strip()
                if mgender and mgender not in {"", "unknown", "n/a", "nan"}:
                    genders.append(mgender)
                text_corpus.append(
                    f"{meta.get('productDisplayName', '')} {meta.get('articleType', '')} {meta.get('usage', '')}"
                )

        dominant_gender = Counter(genders).most_common(1)[0][0] if genders else ""
        primary_color = Counter(colors).most_common(1)[0][0] if colors else ""
        style_tokens = self._extract_style_tokens(" ".join(text_corpus))

        return {
            "detected_categories": categories,
            "dominant_gender": dominant_gender,
            "primary_color": primary_color,
            "style_tokens": style_tokens,
        }
    
    def _find_matches_by_category(self, context: Dict,
                                 complement_category: str,
                                 k: int = 5) -> List[Dict]:
        """
        Find items matching a category using metadata rules.
        
        Rules applied:
        1. Match by category/type
        2. Match by gender (if available)
        3. Optional: match by color palette
        4. Return top-k items
        
        Args:
            detected_item: The original detected item (to extract metadata for matching)
            complement_category: Target category (e.g., "bottom", "shoes")
            k: Number of results to return
        
        Returns:
            List of matching product dicts
        """
        matches = []
        
        if self.metadata_df is None or self.metadata_df.empty:
            return matches
        
        detected_gender = str(context.get("dominant_gender", "")).lower()
        detected_color = str(context.get("primary_color", "")).lower()
        style_tokens = context.get("style_tokens", [])

        candidate_items = self.metadata_df[
            self.metadata_df["normalized_type"].str.lower() == complement_category
        ]
        
        if candidate_items.empty:
            return matches
        
        # Rule 1: Match by gender (if available and not "unknown")
        if detected_gender and detected_gender != "unknown":
            gender_match = (
                (candidate_items["gender"].str.lower() == detected_gender) |
                (candidate_items["gender"].isna())
            )
            candidate_items = candidate_items[gender_match]
        
        if candidate_items.empty:
            return matches

        scored_rows: List[Tuple[float, Dict]] = []
        for _, row in candidate_items.iterrows():
            row_color = str(row.get("baseColour", "")).lower().strip()
            row_text = f"{row.get('productDisplayName', '')} {row.get('articleType', '')} {row.get('usage', '')}".lower()

            score = 0.0
            color_score, color_reason = self._color_compatibility(detected_color, row_color)
            score += color_score

            style_overlap = sum(1 for t in style_tokens if t and t in row_text)
            score += min(1.5, 0.5 * style_overlap)

            if detected_gender and detected_gender != "unknown":
                row_gender = str(row.get("gender", "")).lower().strip()
                if row_gender == detected_gender:
                    score += 1.0

            scored_rows.append((score, {
                "row": row,
                "color_reason": color_reason,
                "style_overlap": style_overlap,
            }))

        scored_rows.sort(key=lambda x: x[0], reverse=True)
        matching_rows = [item for _, item in scored_rows[:k]]
        
        # Format matched items
        for idx, scored in enumerate(matching_rows):
            row = scored["row"]
            product_info = {
                "product_id": str(row.get("id", f"prod_{idx}")),
                "product_name": str(row.get("productDisplayName", "Unknown")),
                "category": str(row.get("masterCategory", "Unknown")),
                "color": str(row.get("baseColour", "Unknown")),
                "gender": str(row.get("gender", "Any")),
                "reason": self._generate_reason(
                    complement_category,
                    context,
                    scored["color_reason"],
                    scored["style_overlap"],
                )
            }
            matches.append(product_info)
        
        return matches
    
    def _generate_reason(self,
                        complement_cat: str,
                        context: Dict,
                        color_reason: str,
                        style_overlap: int) -> str:
        """
        Generate a human-readable reason for the recommendation.
        
        Args:
            complement_cat: Category being recommended (e.g., "bottom")
            original_cat: Original detected category (e.g., "top")
            gender: Gender if available
        
        Returns:
            Recommendation explanation string
        """
        detected_categories = [c for c in context.get("detected_categories", []) if c in {"top", "bottom", "shoes"}]
        if detected_categories:
            based_on = "/".join(sorted(set(detected_categories)))
        else:
            based_on = "outfit"

        style_phrase = "with matching style details" if style_overlap > 0 else "with compatible silhouette"
        return f"Pairs well with your {based_on}: {color_reason} and {style_phrase}."

    def _normalize_item_type(self, article_type: str, sub_category: str, product_name: str) -> str:
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

    def _extract_style_tokens(self, text: str) -> List[str]:
        """
        Extract lightweight style tokens from product text.
        """
        text = str(text or "").lower()
        whitelist = {
            "casual", "formal", "sport", "sports", "running", "printed", "solid", "denim",
            "stripe", "striped", "graphic", "polo", "party", "ethnic", "classic", "street",
        }
        tokens = [t for t in re.findall(r"[a-z]+", text) if t in whitelist]
        return list(dict.fromkeys(tokens))[:8]

    def _color_compatibility(self, base_color: str, candidate_color: str) -> Tuple[float, str]:
        """
        Score color compatibility for simple harmony/contrast rules.
        """
        base = (base_color or "").lower().strip()
        cand = (candidate_color or "").lower().strip()

        if not base or base in {"unknown", "n/a", "nan"}:
            return 0.5, "balanced color pairing"
        if not cand or cand in {"unknown", "n/a", "nan"}:
            return 0.2, "neutral color balance"

        neutrals = {"black", "white", "grey", "gray", "navy blue", "navy", "beige", "brown", "khaki"}
        if base == cand:
            return 1.2, "tonal color harmony"
        if base in neutrals or cand in neutrals:
            return 1.1, "neutral balance"

        complementary = {
            "blue": {"orange", "mustard", "yellow"},
            "navy blue": {"orange", "tan", "beige", "white"},
            "green": {"pink", "beige", "white"},
            "red": {"black", "white", "blue"},
            "orange": {"blue", "navy blue", "black"},
            "pink": {"grey", "white", "green"},
            "yellow": {"blue", "navy blue", "black"},
        }

        if cand in complementary.get(base, set()) or base in complementary.get(cand, set()):
            return 1.3, "intentional contrast"

        return 0.8, "color harmony"
