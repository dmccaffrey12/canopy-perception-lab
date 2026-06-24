import streamlit as st
import numpy as np
import pandas as pd
import json
from ultralytics import YOLO
from video_handler import get_frame
from stress_engine import apply_environmental_stress

# Configure page settings
st.set_page_config(page_title="Canopy Perception Lab", page_icon="🔬", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .reportview-container {
        background: #0f1116;
    }
    .regression-banner {
        padding: 1.2rem;
        background-color: #d32f2f;
        color: white;
        font-weight: bold;
        font-size: 1.2rem;
        border-radius: 0.4rem;
        text-align: center;
        margin-bottom: 1.2rem;
        border: 2px solid #b71c1c;
    }
    .ok-banner {
        padding: 1.2rem;
        background-color: #2e7d32;
        color: white;
        font-weight: bold;
        font-size: 1.2rem;
        border-radius: 0.4rem;
        text-align: center;
        margin-bottom: 1.2rem;
        border: 2px solid #1b5e20;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🔬 Canopy Perception Lab")
st.subheader("Autonomous Vehicle Perception Oracle & Edge-Case Havoc Dashboard")

# 1. Cache the YOLO models
@st.cache_resource
def load_yolo_model(model_name):
    weights = "yolov8n.pt" if "Nano" in model_name else "yolov8s.pt"
    return YOLO(weights)

# 2. Sidebar Configuration
with st.sidebar:
    st.header("1. Input Selector")
    input_type = st.radio(
        "Choose Video Source:",
        ["Autonomous Vehicle Benchmarks", "Upload Custom Video", "YouTube URL"]
    )
    
    video_source = None
    
    if input_type == "Autonomous Vehicle Benchmarks":
        benchmark_choice = st.selectbox(
            "Select Benchmark Sequence:",
            [
                "KITTI Driving Sequence (Standard Day)",
                "Urban Intersection Sequence (Dense Street)",
                "Driver Environment (Cabin View)"
            ]
        )
        if "KITTI" in benchmark_choice:
            video_source = "https://github.com/intel-iot-devkit/sample-videos/raw/master/car-detection.mp4"
        elif "Urban" in benchmark_choice:
            video_source = "https://github.com/AlexeyAB/darknet/raw/master/data/video.mp4"
        else:
            video_source = "https://github.com/intel-iot-devkit/sample-videos/raw/master/driver-action-recognition.mp4"
            
    elif input_type == "Upload Custom Video":
        video_source = st.file_uploader("Upload local video file (.mp4, .avi):", type=["mp4", "avi"])
        
    else:
        video_source = st.text_input("Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")
        
    st.header("2. Timeline Scrubber")
    frame_pct = st.slider("Video Position", min_value=0.0, max_value=100.0, value=50.0, step=1.0) / 100.0
    
    st.header("3. Model Config")
    selected_model_name = st.selectbox("Select Model Size:", ["YOLOv8 Nano (Fast)", "YOLOv8 Small (Accurate)"])
    
    st.header("4. Havoc Controls")
    fog_intensity = st.slider("Gaussian Fog", min_value=0.0, max_value=10.0, value=0.0, step=0.5)
    rain_intensity = st.slider("Torrential Rain", min_value=0.0, max_value=10.0, value=0.0, step=0.5)
    shadow_intensity = st.slider("Canopy Shadow", min_value=0.0, max_value=10.0, value=0.0, step=0.5)

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

# Load YOLO model
model = load_yolo_model(selected_model_name)

# 3. Main Ingestion & Processing
if video_source:
    try:
        # Load frame from source
        with st.spinner("Extracting frame from source..."):
            pristine_frame, is_youtube_fallback = get_frame(video_source, frame_pct)
            
        if is_youtube_fallback:
            st.warning("⚠️ YouTube blocked direct video streaming from our cloud server (HTTP 403). Fell back to the high-resolution video thumbnail. Run locally to extract exact middle frames.")
            
        # Establish Ground Truth
        with st.spinner("Analyzing Ground Truth (Pristine Frame)..."):
            results_pristine = model(pristine_frame, verbose=False)
            
        # Apply Havoc Filters
        corrupted_frame = apply_environmental_stress(
            pristine_frame, 
            fog_intensity, 
            rain_intensity, 
            shadow_intensity
        )
        
        # Evaluate Stressed Frame
        with st.spinner("Analyzing Stressed Frame..."):
            results_corrupted = model(corrupted_frame, verbose=False)
            
        # Parsing Detections
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
            
        # Perform comparison
        comparison_rows = []
        regression_detected = False
        
        for idx, p_det in enumerate(pristine_dets):
            best_iou = 0.0
            best_match = None
            
            for c_det in corrupted_dets:
                if p_det['class'] == c_det['class']:
                    iou = calculate_iou(p_det['bbox'], c_det['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_match = c_det
                        
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
                    "Pristine Conf": p_conf,
                    "Corrupted Conf": c_conf,
                    "Drop (Absolute)": conf_drop,
                    "Status": status
                })
            else:
                regression_detected = True
                comparison_rows.append({
                    "Target ID": f"{p_det['class']} #{idx + 1}",
                    "Class": p_det['class'],
                    "Pristine Conf": p_det['conf'],
                    "Corrupted Conf": 0.0,
                    "Drop (Absolute)": p_det['conf'],
                    "Status": "❌ CRITICAL REGRESSION: TARGET DROPPED"
                })
                
        # Tabbed Dashboard Layout
        tab1, tab2, tab3 = st.tabs(["🎥 Live Oracle Testbed", "📈 Degradation Curves", "💾 Export Test Report"])
        
        with tab1:
            # Layout comparison images side-by-side
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Baseline Detections (Pristine)")
                pristine_plotted = results_pristine[0].plot()
                st.image(pristine_plotted, use_container_width=True)
            with col2:
                st.subheader("Havoc Pipeline Detections")
                corrupted_plotted = results_corrupted[0].plot()
                st.image(corrupted_plotted, use_container_width=True)
                
            # Render Oracle Banner
            if not pristine_dets:
                st.warning("⚠️ No objects detected in the pristine frame. Adjust timeline position or select a different demo.")
            else:
                if regression_detected:
                    st.markdown(
                        '<div class="regression-banner">🚨 CRITICAL REGRESSION DETECTED: MODEL CONFIDENCE DROPPED OR TARGET DROPPED</div>', 
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="ok-banner">✅ ORACLE PASS: NO SIGNIFICANT PERCEPTION REGRESSIONS</div>', 
                        unsafe_allow_html=True
                    )
                    
                # Format comparison DataFrame
                df_disp = pd.DataFrame(comparison_rows).copy()
                df_disp["Pristine Conf"] = df_disp["Pristine Conf"].map(lambda x: f"{x:.1%}")
                df_disp["Corrupted Conf"] = df_disp["Corrupted Conf"].map(lambda x: f"{x:.1%}")
                df_disp["Drop (Absolute)"] = df_disp["Drop (Absolute)"].map(lambda x: f"{x:+.1%}")
                st.table(df_disp)
                
        with tab2:
            st.subheader("Performance Degradation Sweep")
            st.write("This sweeps the noise mixture from 0% to 100% intensity to plot the model's breakdown threshold.")
            
            if not pristine_dets:
                st.info("Please select a frame with active detections to view curves.")
            else:
                if st.button("📊 Calculate Degradation Curve"):
                    sweep_steps = [0.0, 0.25, 0.5, 0.75, 1.0]
                    sweep_confs = []
                    
                    with st.spinner("Sweeping noise levels..."):
                        for step in sweep_steps:
                            # Apply scaled noise
                            swept_frame = apply_environmental_stress(
                                pristine_frame,
                                fog_intensity * step,
                                rain_intensity * step,
                                shadow_intensity * step
                            )
                            # Inference
                            sweep_res = model(swept_frame, verbose=False)
                            sweep_dets = []
                            for box in sweep_res[0].boxes:
                                sweep_dets.append({
                                    'class': model.names[int(box.cls[0])],
                                    'conf': float(box.conf[0]),
                                    'bbox': box.xyxy[0].tolist()
                                })
                                
                            # Calculate average confidence of ground-truth matches
                            confs = []
                            for p_det in pristine_dets:
                                best_iou = 0.0
                                match_conf = 0.0
                                for s_det in sweep_dets:
                                    if p_det['class'] == s_det['class']:
                                        iou = calculate_iou(p_det['bbox'], s_det['bbox'])
                                        if iou > best_iou:
                                            best_iou = iou
                                            match_conf = s_det['conf']
                                            
                                if best_iou > 0.3:
                                    confs.append(match_conf)
                                else:
                                    confs.append(0.0) # Target dropped
                                    
                            sweep_confs.append(np.mean(confs) if confs else 0.0)
                            
                    # Plot Chart
                    chart_data = pd.DataFrame({
                        "Noise Mixture Level (%)": [f"{int(s*100)}%" for s in sweep_steps],
                        "Average Model Confidence": sweep_confs
                    }).set_index("Noise Mixture Level (%)")
                    
                    st.line_chart(chart_data)
                    st.success("Analysis complete!")
                    
        with tab3:
            st.subheader("Export Diagnostics Report")
            st.write("Export the current configuration and regression log for archiving or audit purposes.")
            
            report_data = {
                "framework": "Canopy Perception Lab",
                "model_version": selected_model_name,
                "video_source_type": input_type,
                "video_position_pct": f"{frame_pct:.1%}",
                "stress_parameters": {
                    "fog": fog_intensity,
                    "rain": rain_intensity,
                    "shadow": shadow_intensity
                },
                "summary": {
                    "pristine_detections_count": len(pristine_dets),
                    "corrupted_detections_count": len(corrupted_dets),
                    "regression_detected": regression_detected
                },
                "detections": comparison_rows
            }
            
            # Display JSON
            st.json(report_data)
            
            # Download Button
            json_str = json.dumps(report_data, indent=4)
            st.download_button(
                label="📥 Download JSON Report",
                data=json_str,
                file_name="canopy_perception_report.json",
                mime="application/json"
            )
            
    except Exception as e:
        st.error(f"Error processing video source: {e}")
        st.info("Please make sure your input source is valid and reachable.")
else:
    st.info("Please configure and select a video source in the sidebar to load the framework.")
