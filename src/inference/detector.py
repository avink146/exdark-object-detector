"""
Inference engine and detector for Low-Light Object Detection.
Loads YOLOv8 fine-tuned models, executes memory-efficient predictions,
and formats outputs into [x_min, y_min, width, height] format.
"""

import time
from pathlib import Path
from typing import Union, List, Dict, Any, Optional
import numpy as np
from PIL import Image

CLASS_NAMES = [
    'Bicycle', 'Boat', 'Bottle', 'Bus', 'Cat', 'Cup', 
    'Motorbike', 'People', 'Table', 'car', 'chair', 'dog'
]

class ExDarkDetector:
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        """
        Initializes the low-light object detector.
        Args:
            model_path: Path to .pt model weights. If None, checks models/best.pt then yolov8n.pt.
            device: 'cpu' or 'cuda' (falls back gracefully to cpu).
        """
        self.device = device
        self.classes = CLASS_NAMES

        # Select model weight path
        if model_path and Path(model_path).exists():
            self.weights_path = Path(model_path).resolve()
        elif Path("models/best.pt").exists():
            self.weights_path = Path("models/best.pt").resolve()
        elif Path("models/yolov8n.pt").exists():
            self.weights_path = Path("models/yolov8n.pt").resolve()
        else:
            self.weights_path = "yolov8n.pt"  # Ultralytics will auto-download pretrained weights

        print(f"Loading ExDark detector weights from: {self.weights_path} on {self.device}")
        from ultralytics import YOLO
        self.model = YOLO(str(self.weights_path))

    def predict(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int = 416
    ) -> Dict[str, Any]:
        """
        Runs object detection on a low-light RGB image.
        Returns:
            Dict containing:
            - detections: List of dicts with:
                - box_xywh: [x_min, y_min, width, height] (Official Problem Spec)
                - class_id: int
                - class_name: str
                - confidence: float
            - num_detections: int
            - annotated_image: PIL Image with rendered visual boxes
            - inference_time_ms: float
        """
        start_t = time.perf_counter()

        results = self.model.predict(
            source=image_input,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            device=self.device,
            verbose=False
        )

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        res = results[0]

        detections = []
        boxes_data = res.boxes

        if boxes_data is not None and len(boxes_data) > 0:
            xyxy = boxes_data.xyxy.cpu().numpy()
            xywh = boxes_data.xywh.cpu().numpy()
            confs = boxes_data.conf.cpu().numpy()
            clss = boxes_data.cls.cpu().numpy().astype(int)

            for i in range(len(clss)):
                cid = clss[i]
                cname = self.classes[cid] if cid < len(self.classes) else (res.names.get(cid, f"class_{cid}"))
                x1, y1, x2, y2 = xyxy[i]
                w = x2 - x1
                h = y2 - y1

                detections.append({
                    "box_xywh": [round(float(x1), 2), round(float(y1), 2), round(float(w), 2), round(float(h), 2)],
                    "box_xyxy": [round(float(x1), 2), round(float(y1), 2), round(float(x2), 2), round(float(y2), 2)],
                    "class_id": int(cid),
                    "class_name": str(cname),
                    "confidence": round(float(confs[i]), 4)
                })

        # Render annotated image
        annotated_bgr = res.plot()
        annotated_rgb = annotated_bgr[..., ::-1]
        annotated_pil = Image.fromarray(annotated_rgb)

        return {
            "detections": detections,
            "num_detections": len(detections),
            "annotated_image": annotated_pil,
            "inference_time_ms": round(elapsed_ms, 2)
        }
