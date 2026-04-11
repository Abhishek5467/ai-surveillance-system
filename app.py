import streamlit as st
import cv2
import torch
import torch.nn as nn
import numpy as np
import os
from ultralytics import YOLO
from openai import OpenAI

# 1. SETUP
st.set_page_config(page_title="AI Surveillance", layout="wide")

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# 2. MODEL
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

# 3. LOAD RESOURCES
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_all_models():
    # Load AE
    ae = Conv3DAutoencoder()
    if os.path.exists("model.pth"):
        ae.load_state_dict(torch.load("model.pth", map_location=device))
    ae.to(device).eval()
    
    # Load YOLO
    yolo = YOLO("yolov8n.pt")
    yolo.to(device)
    
    return ae, yolo

ae_model, yolo_model = load_all_models()

# 4. HELPERS
def get_mask_from_error(error_seq, frame_idx=8):
    error_seq = error_seq.cpu().numpy()
    start, end = max(0, frame_idx - 2), min(len(error_seq), frame_idx + 3)
    error = np.mean(error_seq[start:end], axis=0)[0]
    error = np.maximum(error - np.mean(error_seq, axis=0)[0], 0)
    error_norm = (error - error.min()) / (error.max() - error.min() + 1e-8)
    
    mask = (error_norm >= np.sort(error_norm.flatten())[-int(0.02 * error_norm.size)]).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if labels.max() > 0:
        mask = (labels == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    return cv2.dilate(mask, np.ones((11,11), np.uint8))

def detect_objects(frame_3ch, mask):
    masked_frame = frame_3ch.copy()
    masked_frame[mask == 0] = 0
    results = yolo_model(masked_frame, verbose=False)
    return [yolo_model.names[int(b.cls[0])] for r in results for b in r.boxes]

def analyze(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < 16:
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(cv2.resize(frame, (224, 224)), cv2.COLOR_BGR2GRAY) / 255.0)
    cap.release()

    if len(frames) < 16: return None, None, None

    seq = torch.tensor(np.array(frames), dtype=torch.float32).unsqueeze(0).unsqueeze(2).to(device)
    with torch.no_grad():
        recon = ae_model(seq)
    
    mask = get_mask_from_error((seq - recon)**2[0], 8)
    orig_gray = (seq[0, 8, 0].cpu().numpy() * 255).astype(np.uint8)
    orig_3ch = cv2.cvtColor(orig_gray, cv2.COLOR_GRAY2BGR) # CRITICAL FIX

    detected = detect_objects(orig_3ch, mask)
    valid = [obj for obj in detected if obj in ["bicycle", "skateboard", "car", "truck"]]
    
    explanation = "Anomaly detected!" if valid else "No clear anomaly found."
    return orig_gray, mask, explanation

# 5. UI
st.title("🛡️ AI Surveillance System")
uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi"])

if uploaded_file:
    with open("temp.mp4", "wb") as f:
        f.write(uploaded_file.read())
    with st.spinner("Analyzing..."):
        orig, mask, explanation = analyze("temp.mp4")
    
    if orig is not None:
        c1, c2 = st.columns(2)
        c1.image(orig, caption="Original")
        c2.image(mask * 255, caption="Mask")
        st.info(explanation)
    else:
        st.error("Processing failed. Check video length/format.")