import streamlit as st
import numpy as np
import pandas as pd
from ultralytics import YOLO
from video_handler import get_middle_frame
from stress_engine import apply_environmental_stress

# Configure page settings
st.set_page_config(page_title="Canopy Perception Lab", page_icon="🎥", layout="wide")

# Custom styling for professional AI layout
st.markdown("""
    <style>
    .reportview-container {
        background: #0f1116;
    }
    .regression-banner {
        padding: 1.5rem;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        font-size: 1.25rem;
        border-radius: 0.5rem;
        text-align: center;
        margin-bottom: 1.5rem;
        border: 2px solid #b30000;
    }
    .ok-banner {
        padding: 1.5rem;
        background-color: #2e7d32;
        color: white;
        font-weight: bold;
        font-size: 1.25rem;
        border-radius: 0.5rem;
        text-align: center;
        margin-bottom: 1.5rem;
        border: 2px solid #1b5e20;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🔬 Canopy Perception Lab")
st.subheader("Differential Oracle for Computer Vision Model Stress-Testing")

# 1. Load the YOLO model
@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")

try:
    model = load_yolo_model()
except Exception as e:
    st.error(f"Error loading YOLOv8 model: {e}")
    st.stop()

# 2. Sidebar Configuration
with st.sidebar:
    st.header("1. Input Stream")
    youtube_url = st.text_input(
        "YouTube Video URL:",
        placeholder="https://www.youtube.com/watch?v=..."
    )
    
    st.header("2. Environmental Havoc Sliders")
    fog_intensity = st.slider("Gaussian Fog Intensity", min_value=0.0, max_value=10.0, value=0.0, step=0.5)
    rain_intensity = st.slider("Torrential Rain Intensity", min_value=0.0, max_value=10.0, value=0.0, step=0.5)
    shadow_intensity = st.slider("Canopy Shadow Intensity", min_value=0.0, max_value=10.0, value=0.0, step=0.5)

# Bounding box IoU calculation
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area

# 3. Main processing pipeline
if youtube_url:
    try:
        # Frame extraction
        with st.spinner("Extracting stream middle frame..."):
            pristine_frame = get_middle_frame(youtube_url)
            
        # Running Inference on Ground Truth
        with st.spinner("Establishing Ground Truth (Pristine Frame)..."):
            results_pristine = model(pristine_frame, verbose=False)
            
        # Applying Stress
        corrupted_frame = apply_environmental_stress(
            pristine_frame, 
            fog_intensity, 
            rain_intensity, 
            shadow_intensity
        )
        
        # Running Inference on Corrupted Frame
        with st.spinner("Evaluating Stressed Pipeline (Corrupted Frame)..."):
            results_corrupted = model(corrupted_frame, verbose=False)
            
        # Render visual side-by-side columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("Baseline Ground Truth (Pristine)")
            # Plot detections
            pristine_plotted = results_pristine[0].plot()
            st.image(pristine_plotted, use_container_width=True)
            
        with col2:
            st.header("Corrupted Evaluation")
            corrupted_plotted = results_corrupted[0].plot()
            st.image(corrupted_plotted, use_container_width=True)
            
        # 4. Oracle Comparison Logic
        pristine_dets = []
        for box in results_pristine[0].boxes:
            pristine_dets.append({
                'class': model.names[int(box.cls[0])],
                'conf': float(box.conf[0]),
                'bbox': box.xyxy[0].tolist()
            })
            
        corrupted_dets = []
        for box in results_corrupted[0].boxes:
            corrupted_dets.append({
                'class': model.names[int(box.cls[0])],
                'conf': float(box.conf[0]),
                'bbox': box.xyxy[0].tolist()
            })
            
        if not pristine_dets:
            st.warning("⚠️ No objects detected in the baseline (Pristine) frame. Oracle comparison skipped.")
        else:
            comparison_rows = []
            regression_detected = False
            
            for idx, p_det in enumerate(pristine_dets):
                best_iou = 0.0
                best_match = None
                
                # Match to the corrupted detection of same class with best IoU
                for c_det in corrupted_dets:
                    if p_det['class'] == c_det['class']:
                        iou = calculate_iou(p_det['bbox'], c_det['bbox'])
                        if iou > best_iou:
                            best_iou = iou
                            best_match = c_det
                            
                # Evaluation thresholds
                if best_match and best_iou > 0.3:
                    p_conf = p_det['conf']
                    c_conf = best_match['conf']
                    conf_drop = p_conf - c_conf
                    
                    status = "OK"
                    if conf_drop > 0.20:
                        status = "⚠️ Regression: Confidence Dropped"
                        regression_detected = True
                        
                    comparison_rows.append({
                        "Target ID": f"{p_det['class']} #{idx + 1}",
                        "Class": p_det['class'],
                        "Pristine Conf": f"{p_conf:.1%}",
                        "Corrupted Conf": f"{c_conf:.1%}",
                        "Drop (Absolute)": f"{conf_drop:+.1%}",
                        "Status": status
                    })
                else:
                    # Target completely lost
                    regression_detected = True
                    comparison_rows.append({
                        "Target ID": f"{p_det['class']} #{idx + 1}",
                        "Class": p_det['class'],
                        "Pristine Conf": f"{p_det['conf']:.1%}",
                        "Corrupted Conf": "0.0% (Not Detected)",
                        "Drop (Absolute)": "+100.0%",
                        "Status": "❌ CRITICAL REGRESSION: TARGET DROPPED"
                    })
            
            # Dynamic Regression Warnings
            st.header("📊 Oracle Evaluation Results")
            if regression_detected:
                st.markdown(
                    '<div class="regression-banner">🚨 CRITICAL REGRESSION DETECTED: TARGET DROPPED OR CONFIDENCE PLUMMETED</div>', 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="ok-banner">✅ ORACLE PASS: NO SIGNIFICANT PERCEPTION REGRESSIONS</div>', 
                    unsafe_allow_html=True
                )
                
            # Render comparison details table
            df = pd.DataFrame(comparison_rows)
            st.table(df)
            
    except Exception as e:
        st.error(f"Failed to process video: {e}")
        st.info("Ensure the URL is a valid, public YouTube video.")
else:
    st.info("Please enter a YouTube video URL in the sidebar to initiate the evaluation.")
