import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()

m1 = """# Notebook 06: Comprehensive Multi-Model Validation Evaluation

## 1. Experimental Overview & Model Lineage
In accordance with strict scientific evaluation protocols, all model comparisons are conducted on the **fixed ExDark validation split (1,469 images)**. The protected test split (734 images) is strictly sequestered until final model freeze.

Four distinct experimental configurations have been executed and logged:
1. **EXP-001 (Baseline):** 10% data (514 images), 3 epochs, Linear LR decay.
2. **EXP-002-PRELIM:** 15% data (771 images), 6 epochs, Cosine LR (*Confounded*).
3. **EXP-002-CONTROLLED:** 10% data (514 images), 3 epochs, Cosine LR (*Strict 1-variable control*).
4. **EXP-005-FULL-DATA:** 100% data (5,142 images), 5 epochs, Cosine LR (*Full dataset scaling*).
"""

c1 = """import sys
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path('.').resolve()
if not (PROJECT_ROOT / 'reports').exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

print(f'Project root: {PROJECT_ROOT}')
"""

m2 = """## 2. Multi-Experiment Performance Registry
Loaded directly from `reports/experiment_registry.json`."""

c2 = """reg_file = PROJECT_ROOT / 'reports' / 'experiment_registry.json'
with open(reg_file, 'r', encoding='utf-8') as f:
    registry_data = json.load(f)

reg_df = pd.DataFrame(registry_data)
display_cols = [
    'experiment_id', 'scientific_validity', 'train_count', 'fraction',
    'epochs', 'cos_lr', 'best_val_mAP50', 'best_val_mAP50_95',
    'best_precision', 'best_recall', 'training_time', 'checkpoint'
]
summary_df = reg_df[display_cols].rename(columns={
    'experiment_id': 'Experiment ID',
    'scientific_validity': 'Scientific Validity',
    'train_count': 'Train Imgs',
    'fraction': 'Fraction',
    'epochs': 'Epochs',
    'cos_lr': 'Cosine LR',
    'best_val_mAP50': 'Val mAP@50',
    'best_val_mAP50_95': 'Val mAP@50-95',
    'best_precision': 'Val Precision',
    'best_recall': 'Val Recall',
    'training_time': 'Time (s)',
    'checkpoint': 'Checkpoint'
})
display(summary_df)
"""

m3 = """## 3. Comparative Metric Visualizations Across Experiments"""

c3 = """plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

exp_labels = ['EXP-001\\n(Baseline)', 'EXP-002-PRELIM\\n(Confounded)', 'EXP-002-CTRL\\n(Controlled)', 'EXP-005-FULL\\n(Full Data)']
colors = ['#94a3b8', '#cbd5e1', '#64748b', '#4f46e5']

# mAP50
map50_vals = reg_df['best_val_mAP50'].tolist()
bars = axes[0].bar(exp_labels, map50_vals, color=colors, width=0.55)
axes[0].set_title('Validation mAP@50 by Experiment', fontweight='bold', fontsize=12)
axes[0].set_ylabel('mAP@50')
axes[0].set_ylim(0, 0.70)
for b in bars:
    h = b.get_height()
    axes[0].text(b.get_x() + b.get_width()/2, h + 0.015, f'{h:.4f}', ha='center', fontweight='bold')

# Recall
recall_vals = reg_df['best_recall'].tolist()
bars = axes[1].bar(exp_labels, recall_vals, color=['#94a3b8', '#cbd5e1', '#64748b', '#06b6d4'], width=0.55)
axes[1].set_title('Validation Recall by Experiment', fontweight='bold', fontsize=12)
axes[1].set_ylabel('Recall')
axes[1].set_ylim(0, 0.70)
for b in bars:
    h = b.get_height()
    axes[1].text(b.get_x() + b.get_width()/2, h + 0.015, f'{h:.4f}', ha='center', fontweight='bold')

# Precision
prec_vals = reg_df['best_precision'].tolist()
bars = axes[2].bar(exp_labels, prec_vals, color=['#94a3b8', '#cbd5e1', '#64748b', '#10b981'], width=0.55)
axes[2].set_title('Validation Precision by Experiment', fontweight='bold', fontsize=12)
axes[2].set_ylabel('Precision')
axes[2].set_ylim(0, 0.90)
for b in bars:
    h = b.get_height()
    axes[2].text(b.get_x() + b.get_width()/2, h + 0.015, f'{h:.4f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()
"""

m4 = """## 4. Finalist Model Selection Protocol
Under the project's predefined best-model criterion:
- **Primary Metric:** Highest COCO mAP@50 on the fixed validation set.
- **Scientific Validity:** Must not be confounded; must use 100% available training data.
- **Zero Leakage:** Test split was not consulted during model development.

### Selection Decision:
- **Selected Finalist Model:** `EXP-005-FULL-DATA-COSINE`
- **Validation mAP@50:** **`0.5612`**
- **Validation Recall:** **`0.5313`**
- **Validation Precision:** **`0.6321`**
- **Validation mAP@50-95:** **`0.2484`**
- **Checkpoint:** `models/exp_full_001_best.pt`
"""

nb.cells = [
    nbf.v4.new_markdown_cell(m1),
    nbf.v4.new_code_cell(c1),
    nbf.v4.new_markdown_cell(m2),
    nbf.v4.new_code_cell(c2),
    nbf.v4.new_markdown_cell(m3),
    nbf.v4.new_code_cell(c3),
    nbf.v4.new_markdown_cell(m4)
]

out_path = Path("notebooks/06_model_evaluation.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook generated at {out_path}")
