"""
Visualization utilities for low-light object detection.
Provides bounding box drawing, side-by-side comparison, class distribution plotting,
and inspection rendering.
"""

from pathlib import Path
from typing import List, Dict, Union, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CLASS_NAMES = [
    'Bicycle', 'Boat', 'Bottle', 'Bus', 'Cat', 'Cup', 
    'Motorbike', 'People', 'Table', 'car', 'chair', 'dog'
]

# High-visibility color palette for low-light images
COLORS = [
    "#FF3838", "#FF9D97", "#FF701F", "#FFB21D", "#CFD231", "#48F90A",
    "#92CC17", "#3DDB86", "#1A9334", "#00D4BB", "#2C99A8", "#00A2FF"
]

def hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def draw_boxes_on_image(
    image: Image.Image,
    boxes: List[List[float]],
    class_ids: List[int],
    scores: Optional[List[float]] = None,
    class_names: List[str] = CLASS_NAMES,
    box_format: str = "xywh_norm" # 'xywh_norm', 'xywh_abs', 'xyxy_abs'
) -> Image.Image:
    """
    Draws bounding boxes, class labels, and confidence scores on an image.
    Supports low-light images with bright, high-contrast outlines.
    """
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    w_img, h_img = img.size

    for idx, box in enumerate(boxes):
        cls_id = int(class_ids[idx])
        cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
        color = COLORS[cls_id % len(COLORS)]

        if box_format == "xywh_norm":
            xc, yc, bw, bh = box
            x1 = (xc - bw / 2) * w_img
            y1 = (yc - bh / 2) * h_img
            x2 = (xc + bw / 2) * w_img
            y2 = (yc + bh / 2) * h_img
        elif box_format == "xywh_abs":
            # [x_min, y_min, width, height]
            x1, y1, bw, bh = box
            x2 = x1 + bw
            y2 = y1 + bh
        elif box_format == "xyxy_abs":
            x1, y1, x2, y2 = box
        else:
            raise ValueError(f"Unknown box format: {box_format}")

        # Clamp coordinates
        x1 = max(0, min(w_img - 1, x1))
        y1 = max(0, min(h_img - 1, y1))
        x2 = max(0, min(w_img - 1, x2))
        y2 = max(0, min(h_img - 1, y2))

        # Draw multi-pixel thick box for visibility in dark images
        for offset in range(3):
            draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline=color)

        # Label text
        label = cls_name
        if scores is not None and idx < len(scores):
            label += f" {scores[idx]:.2f}"

        # Draw label background banner
        text_bbox = draw.textbbox((x1, max(0, y1 - 18)), label)
        draw.rectangle([text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2], fill=color)
        draw.text((x1, max(0, y1 - 18)), label, fill="#000000")

    return img
