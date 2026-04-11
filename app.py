import streamlit as st
import cv2
import torch
import torch.nn as nn
import numpy as np
import os
from ultralytics import YOLO
from openai import OpenAI

# 1. SETUP & CONFIG
st.set_page_config(page_title="AI Surveillance System", layout="wide")

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# 2. MODEL ARCHITECTURE
class Conv3DAutoencoder(nn.Module):
    def __init__(self):
        super(Conv3DAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(128), nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=(2, 2, 2), padding=1, output_padding=1),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=(2, 2, 2), padding=1, output_padding=1),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.ConvTranspose3d(32, 1, kernel_size=3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        z = self.encoder(x)
        out = self.decoder(z)
        out = out.permute(0, 2, 1, 3, 4)
        return out

# 3. RESOURCE LOADING
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_resources():
    ae = Conv3DAutoencoder()
    if os.path.exists("model.pth"):
        ae.load_state_dict(torch.load("model.pth", map_location=device))
    ae.to(device).eval()
    yolo = YOLO("yolov8n.pt")
    yolo.to(device)
    return ae, yolo

ae_model, yolo_model = load_resources()

# 4. HELPER FUNCTIONS
def get_mask_from_error(error_seq, frame_idx=8):
    error_seq = error_seq.cpu().numpy()
    start, end = max(0, frame_idx - 2), min(len(error_seq), frame_idx + 3)
    error = np.mean(error_seq[start:end], axis=0)[0]
    error = np.maximum(error - np.mean(error_seq, axis=0)[0], 0)
    error_norm = (error - error.min()) / (error.max() - error.min() + 1e-8)
    
    mask = (error_norm >= np.sort(error_norm.flatten())[-int(0.02 * error_norm.size)]).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        mask = (labels == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    return cv2.dilate(mask, np.ones((11,11), np.uint8))

def extract_anomaly_info(mask, frame_shape):
    h, w = frame_shape
    coords = np.column_stack(np.where(mask == 1))
    if len(coords) == 0: return None
    
    y_mean, x_mean = coords.mean(axis=0)
    region = f"{'top' if y_mean < h/3 else 'bottom' if y_mean > 2*h/3 else 'middle'}-" \
             f"{'left' if x_mean < w/3 else 'right' if x_mean > 2*w/3 else 'center'}"
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    area, width, height = len(coords), x_max - x_min, y_max - y_min
    
    return {
        "region": region, "area": int(area),
        "aspect_ratio": float(width / (height + 1e-5)),
        "size_category": "medium" if area > 400 else "small"
    }

def generate_llm_explanation(info, detected_objects):
    prompt = f"Anomaly in pedestrian walkway: Region: {info['region']}, Objects: {detected_objects}. Rules: Pedestrians are normal. Bicycles/skateboards/cars are anomalies. Format: 'A <object> is detected in the <region>.' No reasoning."
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM error: {str(e)}"

# 5. MAIN PIPELINE
def analyze(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < 16:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(cv2.resize(frame, (224, 224)), cv2.COLOR_BGR2GRAY) / 255.0
        frames.append(gray)
    cap.release()

    if len(frames) < 16: return None, None, None

    seq = torch.tensor(np.array(frames), dtype=torch.float32).unsqueeze(0).unsqueeze(2).to(device)
    with torch.no_grad():
        recon = ae_model(seq)
    
    # Square the error first, THEN take the first batch index [0]
    error_tensor = (seq - recon)**2
    mask = get_mask_from_error(error_tensor[0], 8)
    
    orig_gray = (seq[0, 8, 0].cpu().numpy() * 255).astype(np.uint8)
    orig_3ch = cv2.cvtColor(orig_gray, cv2.COLOR_GRAY2BGR)

    info = extract_anomaly_info(mask, orig_gray.shape)
    
    valid_objects = []
    if info:
        masked_yolo = orig_3ch.copy()
        masked_yolo[mask == 0] = 0
        results = yolo_model(masked_yolo, verbose=False)
        valid_objects = [yolo_model.names[int(b.cls[0])] for r in results for b in r.boxes 
                         if yolo_model.names[int(b.cls[0])] in ["bicycle", "skateboard", "car", "truck"]]

    if info is None: 
        explanation = "No anomaly detected."
    elif not valid_objects: 
        explanation = "Anomaly detected but object unidentified."
    else: 
        explanation = generate_llm_explanation(info, valid_objects)

    return orig_gray, mask, explanation

# 6. UI LOGIC
st.title("🛡️ AI Surveillance: 3D CNN + LLM")

if os.path.exists("assets/intro.mp4"):
    st.video("assets/intro.mp4")

uploaded_file = st.file_uploader("Upload Security Footage", type=["mp4", "avi"])

if uploaded_file:
    with open("temp.mp4", "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("🕵️ AI Analyst is inspecting frames..."):
        orig, mask, explanation = analyze("temp.mp4")

    if orig is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Original")
            st.image(orig)
        with col2:
            st.subheader("Anomaly Mask")
            st.image(mask * 255)
        with col3:
            st.subheader("Detection")
            overlay = cv2.cvtColor(orig, cv2.COLOR_GRAY2RGB)
            overlay[mask == 1] = [255, 0, 0] # Highlight red
            st.image(overlay)
        
        st.success(f"**AI Report:** {explanation}")
    else:
        st.error("Processing failed. Ensure the video is at least 16 frames long.")