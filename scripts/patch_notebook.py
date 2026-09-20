import json
from pathlib import Path

p = Path('notebooks/03_baseline_training.ipynb')
with open(p, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if 'train_low_light_detector(' in src:
            cell['source'] = [
                'from src.training.trainer import train_low_light_detector\n',
                '\n',
                '# Execute fine-tuning\n',
                'train_results = train_low_light_detector(\n',
                '    data_yaml="config/data.yaml",\n',
                '    model_name="yolov8n.pt",\n',
                '    epochs=3,\n',
                '    imgsz=416,\n',
                '    batch=8,\n',
                '    fraction=0.10,\n',
                '    device=TRAIN_DEVICE,\n',
                '    project="runs/detect",\n',
                '    name="train_baseline",\n',
                '    patience=3\n',
                ')\n',
                'print("Training completed successfully.")'
            ]
        elif 'val_model.val(' in src:
            cell['source'] = [
                'val_model = YOLO("models/best.pt")\n',
                'val_metrics = val_model.val(data="config/data.yaml", split="val", imgsz=416, batch=8, device=TRAIN_DEVICE)\n',
                'print(f"Validation mAP@50:    {val_metrics.box.map50:.4f}")\n',
                'print(f"Validation mAP@50-95: {val_metrics.box.map:.4f}")'
            ]
        elif 'evaluate_model(' in src:
            cell['source'] = [
                'from src.evaluation.evaluator import evaluate_model\n',
                '\n',
                'test_eval = evaluate_model(\n',
                '    model_path="models/best.pt",\n',
                '    data_yaml="config/data.yaml",\n',
                '    split="test",\n',
                '    imgsz=416,\n',
                '    batch=8,\n',
                '    device=TRAIN_DEVICE\n',
                ')'
            ]

with open(p, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Notebook configuration updated successfully.')
