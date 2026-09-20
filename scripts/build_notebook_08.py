import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()

m1 = """# Notebook 08: Finalist Model Inference, Latency Benchmarking & Qualitative Detections

## 1. System Objective & Finalist Architecture
This notebook demonstrates real-time and batch inference using the validated finalist checkpoint:
- **Finalist Checkpoint:** `models/exp_full_001_best.pt`
- **Trained Split:** 100% ExDark Training Split (5,142 images, 5 epochs)
- **Validation Metric:** COCO mAP@50 = **`0.5612`** | Recall = **`0.5313`** | Precision = **`0.6321`**
- **Inference Specification:**
  - Input: Low-light RGB Image
  - Output: Bounding boxes `[x_min, y_min, width, height]` + Class Label + Confidence Score
"""

c1 = """import sys
import os
import time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

PROJECT_ROOT = Path('.').resolve()
if not (PROJECT_ROOT / 'reports').exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

print(f'Project root: {PROJECT_ROOT}')
"""

m2 = """## 2. Model Loading & Architecture Verification"""

c2 = """checkpoint_path = PROJECT_ROOT / 'models' / 'exp_full_001_best.pt'
assert checkpoint_path.exists(), f'Missing checkpoint at {checkpoint_path}'

model = YOLO(str(checkpoint_path))
print('Successfully loaded finalist model checkpoint!')
print(f'Model names: {model.names}')
print(f'Total classes: {len(model.names)}')
"""

m3 = """## 3. Qualitative Detections on Low-Light Images
Visualizing detector localization under deep shadow and complex nighttime lighting."""

c3 = """val_img_dir = PROJECT_ROOT / 'dataset' / 'processed' / 'ExDark' / 'valid' / 'images'
sample_images = sorted(list(val_img_dir.glob('*.jpg')))[:4]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, img_p in enumerate(sample_images):
    # Run prediction
    res = model.predict(source=str(img_p), imgsz=416, conf=0.25, device='cpu', verbose=False)[0]
    
    # Plot detections
    plotted_rgb = res.plot()
    plotted_rgb = cv2.cvtColor(plotted_rgb, cv2.COLOR_BGR2RGB)
    
    axes[idx].imshow(plotted_rgb)
    axes[idx].set_title(f'Sample {idx+1}: {img_p.name}\\nDetections: {len(res.boxes)}', fontsize=11, fontweight='bold')
    axes[idx].axis('off')

plt.tight_layout()
plt.show()
"""

m4 = """## 4. Latency Profiling on Multi-Threaded CPU (10 Threads)
Benchmarking preprocess, inference, and postprocess latency across 20 validation images."""

c4 = """import torch
torch.set_num_threads(10)

test_imgs = sorted(list(val_img_dir.glob('*.jpg')))[10:30]
latencies = []

# Warmup
_ = model.predict(source=str(test_imgs[0]), imgsz=416, conf=0.25, device='cpu', verbose=False)

for p in test_imgs:
    t0 = time.time()
    _ = model.predict(source=str(p), imgsz=416, conf=0.25, device='cpu', verbose=False)
    latencies.append((time.time() - t0) * 1000)

mean_latency = np.mean(latencies)
median_latency = np.median(latencies)
p95_latency = np.percentile(latencies, 95)
fps = 1000.0 / mean_latency

latency_summary = {
    'Metric': ['Mean Latency', 'Median Latency', '95th Percentile Latency', 'Inference Throughput (FPS)'],
    'Measured Value': [f'{mean_latency:.1f} ms', f'{median_latency:.1f} ms', f'{p95_latency:.1f} ms', f'{fps:.1f} FPS']
}
display(pd.DataFrame(latency_summary))
"""

m5 = """## 5. Standard Output Format Compliance
Verifying standard specification format: `[x_min, y_min, width, height], class_label, confidence_score`."""

c5 = """sample_pred = model.predict(source=str(sample_images[0]), imgsz=416, conf=0.25, device='cpu', verbose=False)[0]

formatted_detections = []
for box in sample_pred.boxes:
    xywh = box.xywh[0].tolist() # center_x, center_y, width, height
    cls_id = int(box.cls[0].item())
    cls_name = model.names[cls_id]
    conf = float(box.conf[0].item())
    
    # Convert to x_min, y_min, width, height
    cx, cy, w, h = xywh
    xmin = cx - w/2
    ymin = cy - h/2
    
    formatted_detections.append({
        'class': cls_name,
        'confidence': round(conf, 4),
        'bbox_xmin': round(xmin, 1),
        'bbox_ymin': round(ymin, 1),
        'bbox_width': round(w, 1),
        'bbox_height': round(h, 1)
    })

print(f'Total formatted detections for {sample_images[0].name}: {len(formatted_detections)}')
display(pd.DataFrame(formatted_detections))
"""

nb.cells = [
    nbf.v4.new_markdown_cell(m1),
    nbf.v4.new_code_cell(c1),
    nbf.v4.new_markdown_cell(m2),
    nbf.v4.new_code_cell(c2),
    nbf.v4.new_markdown_cell(m3),
    nbf.v4.new_code_cell(c3),
    nbf.v4.new_markdown_cell(m4),
    nbf.v4.new_code_cell(c4),
    nbf.v4.new_markdown_cell(m5),
    nbf.v4.new_code_cell(c5)
]

out_path = Path("notebooks/08_final_inference.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook generated at {out_path}")
