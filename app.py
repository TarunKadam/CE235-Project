import io
import os
import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from PIL import Image
import rasterio
from rasterio.io import MemoryFile

from common import (
    CLASS_NAMES, CLASS_COLORS, CLASS_IDS, ClusterClassifier,
    compute_features, predict_map, colorize, auto_step
)


st.set_page_config(
    page_title="GeoSpatial AI | Urban Feature Classification Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Global Theme & Deep Navy Background */
    .main {
        background-color: #070b12;
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0b111e;
        border-right: 1px solid #1e293b;
    }
    
    /* Animations */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(16px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Hero Section Styling */
    .hero-wrapper {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 40px;
        background: linear-gradient(135deg, #0b111e 0%, #070b12 100%);
        border: 1px solid #1e293b;
        padding: 50px;
        border-radius: 18px;
        margin-top: 10px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.6);
        animation: fadeInUp 0.7s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
    
    .hero-content {
        flex: 1.1;
    }
    
    .hero-visual {
        flex: 0.9;
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.5);
    }

    /* Metric Container Polish */
    div[data-testid="metric-container"] {
        background-color: #0b111e;
        border: 1px solid #1e293b;
        padding: 15px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

MODEL_FILES = {
    "Logistic Regression": "logreg",
    "SVM (RBF)": "svm",
    "Random Forest": "random_forest",
    "K-Means (clustering)": "kmeans",
    "CNN": "cnn",
}

available = {
    k: v for k, v in MODEL_FILES.items()
    if os.path.exists(f"models/{v}.joblib")
    or os.path.exists(f"models/{v}.pt")
}

@st.cache_resource
def load_model(key):
    if key == "cnn":
        import torch
        from cnn_model import SmallCNN
        m = SmallCNN(len(CLASS_IDS))
        m.load_state_dict(torch.load("models/cnn.pt", map_location="cpu"))
        return m.eval()
    return joblib.load(f"models/{key}.joblib")

def load_uploaded_rgb(file):
    filename = file.name.lower()
    if filename.endswith((".tif", ".tiff")):
        with MemoryFile(file.getvalue()) as memfile:
            with memfile.open() as src:
                if src.count >= 3:
                    arr = src.read([1, 2, 3])
                else:
                    arr = np.repeat(src.read(1)[None], 3, axis=0)
                valid_mask = src.dataset_mask() > 0
        valid_mask &= np.any(arr != 0, axis=0)
        if arr.dtype != np.uint8:
            hi = max(float(np.percentile(arr, 99)), 1.0)
            arr = (np.clip(arr / hi, 0, 1) * 255).astype(np.uint8)
        rgb = np.transpose(arr, (1, 2, 0)).copy()
    else:
        rgb = np.array(Image.open(file).convert("RGB"))
        valid_mask = np.any(rgb != 0, axis=2)
    return rgb, valid_mask

def classify(key, rgb):
    model = load_model(key)
    if key == "cnn":
        from cnn_model import predict_map_cnn
        return predict_map_cnn(model, rgb, stride=8)
    h, w, _ = rgb.shape
    step = auto_step(h, w, limit=300_000)
    return predict_map(model, rgb, step=step)

def legend_html():
    """Generates a clean vertical legend block with exact color bindings."""
    html_items = []
    for c in CLASS_IDS:
        r, g, b = CLASS_COLORS[c]
        html_items.append(
            f"<div style='display: flex; align-items: center; margin-bottom: 9px;'>"
            f"<span style='display: inline-block; width: 13px; height: 13px; "
            f"background: rgb({r},{g},{b}); border-radius: 3px; margin-right: 12px; "
            f"box-shadow: 0 0 6px rgba(0,0,0,0.4);'></span>"
            f"<span style='font-size: 13.5px; color: #cbd5e1; font-weight: 500;'>{CLASS_NAMES[c]}</span>"
            f"</div>"
        )
    return "".join(html_items)

tab1, tab2 = st.tabs(["Inference & Prediction", "Model Performance Benchmarking"])

with tab1:
    if not available:
        st.error("⚠️ No trained models found in `./models`. Please execute training scripts first.")
        st.stop()

    with st.sidebar:
        st.markdown("### Configuration Panel")
        model_name = st.selectbox("Select ML Model", list(available))
        
        st.markdown("<hr style='border-color: #1e293b; margin: 15px 0;'>", unsafe_allow_html=True)
        file = st.file_uploader(
            "Upload Satellite Raster",
            type=["jpg", "jpeg", "png", "tif", "tiff"],
            help="Upload GeoTIFF or high-resolution RGB imagery."
        )
        
        st.markdown("<hr style='border-color: #1e293b; margin: 15px 0;'>", unsafe_allow_html=True)
        st.markdown("### Land-Cover Legend")
        st.markdown(legend_html(), unsafe_allow_html=True)
        
        st.markdown("<hr style='border-color: #1e293b; margin: 15px 0;'>", unsafe_allow_html=True)
        with st.expander("System Architecture"):
            st.markdown(
                "This platform evaluates spatial feature representations "
                "using normalized difference indices, texture metrics, and "
                "supervised/unsupervised classification pipelines."
            )

    if file:
        try:
            with st.spinner("Processing raster data and applying band scaling..."):
                rgb, valid_mask = load_uploaded_rgb(file)

            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("Spatial Resolution", f"{rgb.shape[1]} × {rgb.shape[0]} px")
            mcol2.metric("Active Model Engine", model_name)
            mcol3.metric("Valid Pixel Ratio", f"{100 * valid_mask.sum() / valid_mask.size:.1f}%")

            st.markdown("<hr style='border-color: #1e293b; margin: 20px 0;'>", unsafe_allow_html=True)
            with st.spinner(f"Executing spatial inference using {model_name}..."):
                label_map = classify(available[model_name], rgb)
                
            label_map[~valid_mask] = 0
            colored = colorize(label_map)

            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.markdown("#### Input Satellite Image")
                st.image(rgb, use_container_width=True)
            with col_img2:
                st.markdown(f"#### Thematic Land-Cover Map")
                st.image(colored, use_container_width=True)

            st.markdown("<hr style='border-color: #1e293b; margin: 20px 0;'>", unsafe_allow_html=True)
            st.markdown("### Quantitative Land-Cover Composition")

            counts = np.bincount(label_map.ravel(), minlength=max(CLASS_IDS) + 1)
            total_valid_pixels = max(1, valid_mask.sum())

            area = pd.DataFrame({
                "Class": [CLASS_NAMES[c] for c in CLASS_IDS],
                "Area (%)": [100 * counts[c] / total_valid_pixels for c in CLASS_IDS],
                "Pixel Count": [counts[c] for c in CLASS_IDS]
            })

            chart_col, data_col = st.columns([1.5, 1])
            with chart_col:
                st.bar_chart(area.set_index("Class")["Area (%)"], color="#10b981")
            with data_col:
                st.dataframe(area.round(2), hide_index=True, use_container_width=True)

            st.markdown("<hr style='border-color: #1e293b; margin: 20px 0;'>", unsafe_allow_html=True)
            ex_col1, ex_col2 = st.columns([2, 1])
            with ex_col1:
                st.markdown("**Export Options:** Download georeferenced thematic map layers for GIS software integration.")
            with ex_col2:
                buf = io.BytesIO()
                Image.fromarray(colored).save(buf, format="PNG")
                st.download_button(
                    "Download Thematic Map",
                    buf.getvalue(),
                    file_name=f"thematic_map_{available[model_name]}.png",
                    mime="image/png",
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"❌ Error during spatial processing: {e}")
    else:
        st.markdown(
            """
            <div class="hero-wrapper">
                <div class="hero-content">
                    <span style="background: rgba(16, 185, 129, 0.15); color: #34d399; padding: 6px 14px; border-radius: 20px; font-size: 12px; font-weight: 600; border: 1px solid rgba(16, 185, 129, 0.3);">
                        🛰️ Advanced Geospatial Intelligence Engine
                    </span>
                    <h1 style="color: #f8fafc; font-size: 38px; margin-top: 16px; margin-bottom: 12px; font-weight: 800; line-height: 1.2;">
                        An automated satellite analysis platform built for urban intelligence
                    </h1>
                    <p style="color: #94a3b8; line-height: 1.7; font-size: 15.5px; margin-bottom: 25px;">
                    This platform processes high-resolution satellite imagery and spatial data parcels to automatically extract, 
                    classify, and map urban features. Leverage machine learning architectures to generate thematic land-cover maps 
                    and quantitative analytics for sustainable urban planning.
                    </p>
                    <div style="display: flex; gap: 15px; align-items: center;">
                        <span style="color: #34d399; font-weight: 600; font-size: 14px;">👈 Configure models & upload a raster file in the sidebar to begin</span>
                    </div>
                </div>
                <div class="hero-visual">
                    <div style="font-size: 13px; color: #64748b; margin-bottom: 8px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Supported Urban Classes</div>
                    <div style="display: flex; flex-direction: column; gap: 8px;">
                        <div style="background: rgba(220,50,50,0.15); color: #fca5a5; padding: 8px 12px; border-radius: 6px; font-size: 13px; font-weight: 500; border-left: 3px solid #dc3232;">🟩 Vegetation & Green Spaces</div>
                        <div style="background: rgba(120,120,120,0.15); color: #cbd5e1; padding: 8px 12px; border-radius: 6px; font-size: 13px; font-weight: 500; border-left: 3px solid #787878;">🛣️ Road Networks & Transport</div>
                        <div style="background: rgba(40,100,220,0.15); color: #93c5fd; padding: 8px 12px; border-radius: 6px; font-size: 13px; font-weight: 500; border-left: 3px solid #2864dc;">💧 Rivers & Water Bodies</div>
                        <div style="background: rgba(210,180,110,0.15); color: #fde68a; padding: 8px 12px; border-radius: 6px; font-size: 13px; font-weight: 500; border-left: 3px solid #d2b46e;">🏜️ Barren Land & Open Terrain</div>
                        <div style="background: rgba(220,50,50,0.25); color: #f87171; padding: 8px 12px; border-radius: 6px; font-size: 13px; font-weight: 500; border-left: 3px solid #dc3232;">🏢 Built-up & Urban Infrastructure</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)
        st.markdown("### Analysis in four steps")
        st.markdown("<p style='color: #94a3b8; font-size: 14.5px; margin-bottom: 25px;'>A streamlined workflow from raw satellite tiles to actionable urban analytics.</p>", unsafe_allow_html=True)

        steps_data = [
            ("1", "Select ML Model Architecture", "Choose from trained machine learning models like Random Forest, SVM, Logistic Regression, K-Means, or CNN."),
            ("2", "Upload Raster Tile", "Upload your target georeferenced satellite GeoTIFF or standard image file via the configuration panel."),
            ("3", "Run Classification", "The engine extracts spatial spectral and textural features, categorising pixels into distinct urban classes."),
            ("4", "Map & Export", "Examine thematic maps, analyze land-cover area percentages, and export high-res visual assets.")
        ]

        cols = st.columns(4)
        for idx, (num, title, desc) in enumerate(steps_data):
            with cols[idx]:
                st.markdown(
                    f"""
                    <div style="background-color: #0f172a; border: 1px solid #1e293b; padding: 22px; border-radius: 12px; height: 210px; display: flex; flex-direction: column; justify-content: flex-start;">
                        <div style="width: 36px; height: 36px; background: rgba(16, 185, 129, 0.15); color: #34d399; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 14px; margin-bottom: 14px; border: 1px solid rgba(16, 185, 129, 0.3);">{num}</div>
                        <h4 style="color: #f8fafc; font-size: 15px; margin-bottom: 8px; font-weight: 600;">{title}</h4>
                        <p style="color: #94a3b8; font-size: 13px; line-height: 1.5; margin: 0;">{desc}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

with tab2:
    st.markdown("### Test-Set Metrics & Model Benchmarking")
    st.markdown("Comparative performance analysis across trained traditional classifiers and deep learning models on a satellite image of Pune City")
    
    st.info(
        " **Understanding the Metrics:**\n"
        "- **accuracy**: Overall pixel-wise classification correctness.\n"
        "- **macro_f1**: Class-balanced harmonic mean of precision and recall.\n"
        "- **kappa**: Cohen's Kappa coefficient measuring inter-rater agreement beyond chance.\n"
        "- **train_time_s**: Total wall-clock model training duration in seconds."
    )
    
    if os.path.exists("outputs/results.csv"):
        res = pd.read_csv("outputs/results.csv")
        st.dataframe(res.round(4), hide_index=True, use_container_width=True)

        st.markdown("#### Performance Comparison Chart")
        st.bar_chart(res.set_index("model")[["accuracy", "macro_f1", "kappa"]])
    else:
        st.warning("No evaluation results found in `outputs/results.csv`. Run your training pipeline first.")

    if os.path.exists("outputs/confusion.json"):
        st.markdown("<hr style='border-color: #1e293b; margin: 20px 0;'>", unsafe_allow_html=True)
        st.markdown("### Confusion Matrix Inspection")
        st.markdown("<p style='color: #94a3b8; font-size: 13.5px;'>Inspect class-wise prediction confusions (True vs. Predicted) for each model architecture.</p>", unsafe_allow_html=True)
        
        with open("outputs/confusion.json", "r") as f:
            conf = json.load(f)

        if conf:
            pick = st.selectbox("Select Model Architecture", list(conf))
            names = [CLASS_NAMES[c] for c in CLASS_IDS]

            cm = pd.DataFrame(
                conf[pick],
                index=[f"True: {n}" for n in names],
                columns=[f"Pred: {n}" for n in names],
            )

            st.dataframe(
                cm.style.background_gradient(cmap="YlGnBu", axis=None),
                use_container_width=True
            )