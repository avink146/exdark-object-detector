g"""
Streamlit Web Application: Industry-Quality Low-Light Object Detection Dashboard
Built for the ExDark Benchmark with YOLOv8n Transfer Learning.
"""

import sys
import os
import time
from pathlib import Path
from PIL import Image
import pandas as pd
import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.visualization import CLASS_NAMES, COLORS, draw_boxes_on_image
from src.utils.hardware import get_system_info

# Set Streamlit page config
st.set_page_config(
    page_title="ExDark | Low-Light Object Detection",
    page_icon="🔦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern dark-mode aesthetic
st.markdown("""
<style>
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        margin-bottom: 12px;
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-val {
        color: #38bdf8;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading YOLOv8 Low-Light Model...")
def load_detector(model_path: str):
    """Cached loader for the low-light detector."""
    from src.inference.detector import ExDarkDetector
    return ExDarkDetector(model_path=model_path, device="cpu")

# Sidebar
st.sidebar.title("🔦 ExDark Detector")
st.sidebar.markdown("**Low-Light Object Detection System**")
st.sidebar.markdown("---")

app_mode = st.sidebar.radio(
    "Navigation",
    ["🏠 Project Overview", "🎯 Single Image Detection", "📁 Batch Detection", "📊 Model Performance", "ℹ️ Architecture & Hardware"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Model Selection")
available_models = {}
if (PROJECT_ROOT / "models" / "final_best.pt").exists():
    available_models["Frozen Final Model (EXP-005 Full-Data 5-Epoch Cosine)"] = str(PROJECT_ROOT / "models" / "final_best.pt")
if (PROJECT_ROOT / "models" / "exp_full_001_best.pt").exists():
    available_models["EXP-005 Full-Data Checkpoint"] = str(PROJECT_ROOT / "models" / "exp_full_001_best.pt")
if (PROJECT_ROOT / "models" / "exp002_controlled_best.pt").exists():
    available_models["EXP-002-CONTROLLED (Cosine 1-Variable)"] = str(PROJECT_ROOT / "models" / "exp002_controlled_best.pt")
if (PROJECT_ROOT / "models" / "baseline_best.pt").exists():
    available_models["EXP-001 Baseline (Linear LR)"] = str(PROJECT_ROOT / "models" / "baseline_best.pt")
available_models["Pretrained YOLOv8n (MS COCO)"] = "yolov8n.pt"

selected_model_label = st.sidebar.selectbox("Checkpoint", list(available_models.keys()))
selected_model_path = available_models[selected_model_label]

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Inference Settings")
conf_thresh = st.sidebar.slider("Confidence Threshold", min_value=0.05, max_value=0.95, value=0.25, step=0.05)
iou_thresh = st.sidebar.slider("NMS IoU Threshold", min_value=0.10, max_value=0.80, value=0.45, step=0.05)
img_size = st.sidebar.select_slider("Inference Resolution (px)", options=[320, 416, 640], value=416)

# Hardware status badge in sidebar
hw_info = get_system_info()
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Device:** `{hw_info['device'].upper()}`")
st.sidebar.markdown(f"**GPU Memory Target:** ~2.0 GB VRAM")

# -------------------------------------------------------------------------------------------------
# TAB 1: Project Overview
# -------------------------------------------------------------------------------------------------
if app_mode == "🏠 Project Overview":
    st.title("Low-Light Object Detection System")
    st.markdown("### Real-Time Detection in Underexposed & Extreme Night Imagery")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div class="metric-card"><div class="metric-title">Primary Benchmark</div><div class="metric-val">ExDark</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="metric-card"><div class="metric-title">Target Classes</div><div class="metric-val">12</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="metric-card"><div class="metric-title">Primary Metric</div><div class="metric-val">mAP@50</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class="metric-card"><div class="metric-title">Architecture</div><div class="metric-val">YOLOv8n</div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Why Low-Light Detection is Challenging")
    st.markdown("""
    Standard object detectors (trained on well-lit COCO or Pascal VOC datasets) suffer severe performance degradation in dark environments due to:
    - **Extreme Underexposure & Low Dynamic Range:** Crucial structural edges and fine object textures are compressed into low pixel intensities.
    - **Non-Uniform Night Illuminations:** Harsh localized glare (e.g. vehicle headlights, street lamps) juxtaposed against pitch-black shadows.
    - **High Sensor Shot & Read Noise:** Night photography operates at elevated ISO settings, introducing heavy granular noise that confuses feature extractors.
    - **Motion & Optical Blur:** Long camera exposure times lead to significant motion blur of moving subjects (pedestrians, vehicles).
    """)

    st.subheader("Target Classes (ExDark Dataset)")
    class_badges = "".join([f'<span class="badge" style="background-color: {COLORS[i%len(COLORS)]}33; border: 1px solid {COLORS[i%len(COLORS)]}; color: #ffffff;">{name}</span>' for i, name in enumerate(CLASS_NAMES)])
    st.markdown(class_badges, unsafe_allow_html=True)

    st.subheader("Solution Architecture: Transfer Learning")
    st.info("""
    **Transfer Learning Pipeline:**
    COCO Pretrained YOLOv8n (3.2M params) ➔ Low-Light Photometric & Mosaic Augmentations ➔ Fine-Tuned on ExDark (7,345 images) ➔ Low-VRAM Cached Inference Engine.
    """)

# -------------------------------------------------------------------------------------------------
# TAB 2: Single Image Detection
# -------------------------------------------------------------------------------------------------
elif app_mode == "🎯 Single Image Detection":
    st.title("Single Image Detection")
    st.markdown("Upload any low-light, night, or underexposed image to detect objects in real-time.")

    uploaded_file = st.file_uploader("Choose an image (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

    # Sample demo images option
    sample_images_dir = PROJECT_ROOT / "dataset" / "processed" / "ExDark" / "test" / "images"
    use_sample = st.checkbox("Or test with a sample ExDark test image")
    selected_sample = None
    if use_sample and sample_images_dir.exists():
        sample_files = list(sample_images_dir.glob("*.jpg"))[:10]
        if sample_files:
            sample_choice = st.selectbox("Select sample:", [f.name for f in sample_files])
            selected_sample = sample_images_dir / sample_choice

    image_to_process = None
    if uploaded_file is not None:
        image_to_process = Image.open(uploaded_file).convert("RGB")
    elif selected_sample is not None and selected_sample.exists():
        image_to_process = Image.open(selected_sample).convert("RGB")

    if image_to_process is not None:
        detector = load_detector(selected_model_path)

        with st.spinner("Running low-light detection..."):
            res = detector.predict(
                image_input=image_to_process,
                conf_threshold=conf_thresh,
                iou_threshold=iou_thresh,
                imgsz=img_size
            )

        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.markdown("**Original Low-Light Image**")
            st.image(image_to_process, use_container_width=True)
        with col_img2:
            st.markdown(f"**Detected Objects ({res['num_detections']} detected)**")
            st.image(res["annotated_image"], use_container_width=True)

        # Performance summary metrics
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Total Detections", res["num_detections"])
        with m2:
            st.metric("Latency", f"{res['inference_time_ms']:.1f} ms")
        with m3:
            detected_classes = list(set([d["class_name"] for d in res["detections"]]))
            st.metric("Unique Classes", len(detected_classes))

        # Detections table with [x_min, y_min, width, height]
        if res["detections"]:
            st.markdown("### 📋 Detailed Predictions")
            st.caption("Official Bounding Box Format: `[x_min, y_min, width, height]` (pixels)")
            df_data = []
            for d in res["detections"]:
                df_data.append({
                    "Class": d["class_name"],
                    "Confidence": f"{d['confidence'] * 100:.1f}%",
                    "Box [x_min, y_min, w, h]": str(d["box_xywh"])
                })
            st.dataframe(pd.DataFrame(df_data), use_container_width=True)
        else:
            st.warning("No objects detected at current confidence threshold. Try lowering the threshold slider.")

# -------------------------------------------------------------------------------------------------
# TAB 3: Batch Detection
# -------------------------------------------------------------------------------------------------
elif app_mode == "📁 Batch Detection":
    st.title("Batch Low-Light Detection")
    st.markdown("Upload multiple images to run batch inference and inspect aggregate results.")

    uploaded_files = st.file_uploader("Upload low-light images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        detector = load_detector(selected_model_path)
        st.markdown(f"Processing **{len(uploaded_files)}** images...")
        progress_bar = st.progress(0)

        results_list = []
        for i, file in enumerate(uploaded_files):
            img = Image.open(file).convert("RGB")
            res = detector.predict(img, conf_threshold=conf_thresh, iou_threshold=iou_thresh, imgsz=img_size)
            results_list.append({
                "name": file.name,
                "image": img,
                "annotated": res["annotated_image"],
                "count": res["num_detections"],
                "latency": res["inference_time_ms"],
                "classes": ", ".join(list(set([d["class_name"] for d in res["detections"]])))
            })
            progress_bar.progress((i + 1) / len(uploaded_files))

        st.success(f"Batch completed! Processed {len(uploaded_files)} images.")

        # Batch summary table
        summary_df = pd.DataFrame([{
            "Filename": r["name"],
            "Objects Detected": r["count"],
            "Detected Classes": r["classes"] or "None",
            "Latency (ms)": f"{r['latency']:.1f}"
        } for r in results_list])
        st.dataframe(summary_df, use_container_width=True)

        # Gallery
        st.markdown("### 🖼️ Results Gallery")
        cols = st.columns(2)
        for idx, r in enumerate(results_list):
            with cols[idx % 2]:
                st.markdown(f"**{r['name']}** ({r['count']} objects, {r['latency']:.1f} ms)")
                st.image(r["annotated"], use_container_width=True)

# -------------------------------------------------------------------------------------------------
# TAB 4: Model Performance
# -------------------------------------------------------------------------------------------------
elif app_mode == "📊 Model Performance":
    st.title("Model Performance & Benchmark Evaluation")
    st.markdown("Scientific evaluation and multi-experiment comparison on the ExDark benchmark.")

    import json
    
    # Section 1: Multi-Experiment Comparison
    summary_path = PROJECT_ROOT / "reports" / "experiments_summary.json"
    if summary_path.exists():
        st.subheader("🔬 Multi-Experiment Comparison (Validation Benchmark)")
        with open(summary_path, "r") as f:
            exp_data = json.load(f)
        
        exps = exp_data.get("experiments", [])
        if exps:
            exp_table = []
            for e in exps:
                exp_table.append({
                    "ID": e.get("id"),
                    "Experiment Name": e.get("name"),
                    "Resolution": f"{e.get('imgsz')}x{e.get('imgsz')}",
                    "Epochs": e.get("epochs"),
                    "Val mAP@50": e.get("val_mAP50", "N/A"),
                    "Val Recall": e.get("val_recall", "N/A"),
                    "Val Precision": e.get("val_precision", "N/A"),
                    "Status": e.get("status")
                })
            st.dataframe(pd.DataFrame(exp_table), use_container_width=True)
            
        val_chart = PROJECT_ROOT / "reports" / "val_comparison_chart.png"
        if val_chart.exists():
            st.image(str(val_chart), caption="Validation Performance Comparison Across Experiments", use_container_width=True)
            
        st.markdown("---")
    metrics_path = PROJECT_ROOT / "reports" / "test_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class="metric-card"><div class="metric-title">Primary Metric: mAP@50</div><div class="metric-val">{metrics.get('primary_metric_mAP50', 'N/A')}</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card"><div class="metric-title">Secondary Metric: mAP@50-95</div><div class="metric-val">{metrics.get('mAP50_95', 'N/A')}</div></div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card"><div class="metric-title">Overall Precision</div><div class="metric-val">{metrics.get('overall_precision', 'N/A')}</div></div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class="metric-card"><div class="metric-title">Overall Recall</div><div class="metric-val">{metrics.get('overall_recall', 'N/A')}</div></div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Per-Class AP@50 Performance")
        per_class = metrics.get("per_class_mAP50", {})
        if per_class:
            df_classes = pd.DataFrame(list(per_class.items()), columns=["Class", "AP@50"]).sort_values(by="AP@50", ascending=False)
            st.bar_chart(df_classes.set_index("Class"))
            st.dataframe(df_classes, use_container_width=True)
    else:
        st.info("Evaluation metrics report will be populated upon completing the evaluation pipeline.")

    # Show training plots if available
    train_dir = PROJECT_ROOT / "runs" / "detect" / "train_baseline"
    if (train_dir / "results.png").exists():
        st.markdown("### 📈 Training Loss & Metric Convergence Curves")
        st.image(str(train_dir / "results.png"), use_container_width=True)
    if (train_dir / "confusion_matrix.png").exists():
        st.markdown("### 🔲 Confusion Matrix")
        st.image(str(train_dir / "confusion_matrix.png"), use_container_width=True)

# -------------------------------------------------------------------------------------------------
# TAB 5: Architecture & Hardware
# -------------------------------------------------------------------------------------------------
elif app_mode == "ℹ️ Architecture & Hardware":
    st.title("Model Architecture & Hardware Constraints")

    col_arch, col_hw = st.columns(2)
    with col_arch:
        st.subheader("Model Specifications")
        st.markdown("""
        - **Model Architecture:** YOLOv8n (Nano)
        - **Number of Parameters:** ~3.2 Million
        - **Computational FLOPs:** 8.7 GFLOPs @ 640x640
        - **Input Format:** Low-Light RGB Image
        - **Target Classes:** 12 ExDark categories
        - **Transfer Learning Strategy:** Fine-tuning from MS COCO pretrained weights
        - **Augmentations:** HSV-V (brightness), HSV-S (saturation), Mosaic, Random Erasing
        """)

    with col_hw:
        st.subheader("Hardware & Low-VRAM Optimization")
        st.markdown(f"""
        - **Target GPU Budget:** ~2.0 GB VRAM
        - **Detected GPU:** `{hw_info['gpu_name']}`
        - **Host CPU:** `{hw_info['cpu_name']}` ({hw_info['cpu_cores']} logical threads)
        - **Inference Optimization:**
          - Streamlit model caching (`@st.cache_resource`) prevents re-allocations.
          - Low-resolution processing mode (`imgsz=416`) for lightweight footprint.
          - Graceful CPU multi-threaded execution.
        """)
