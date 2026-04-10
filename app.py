import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from glob import glob
import matplotlib.pyplot as plt
import streamlit as st

class Conv3DAutoencoder(nn.Module):

    def __init__(self):
        super(Conv3DAutoencoder,self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(1,32,kernel_size=3,stride=(1,2,2),padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.Conv3d(32,64,kernel_size=3,stride=(2,2,2),padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(64,128,kernel_size=3,stride=(2,2,2),padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(128,64,kernel_size=3,stride=(2,2,2),padding=1,output_padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(64,32,kernel_size=3,stride=(2,2,2),padding=1,output_padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(32,1,kernel_size=3,stride=(1,2,2),padding=1,output_padding=(0,1,1)),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = x.permute(0,2,1,3,4)
        z = self.encoder(x)
        out = self.decoder(z)
        out = out.permute(0,2,1,3,4)
        return out

@st.cache_resource
def load_model():
    model = Conv3DAutoencoder()   # your class
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# =========================
# 🧠 YOUR FUNCTIONS
# =========================

def get_mask_from_error(error_seq, frame_idx=8):

    error_seq = error_seq.cpu().numpy()

    start = max(0, frame_idx - 2)
    end = min(len(error_seq), frame_idx + 3)

    temporal_window = error_seq[start:end]
    error = np.mean(temporal_window, axis=0)[0]

    temporal_mean = np.mean(error_seq, axis=0)[0]
    error = error - temporal_mean
    error = np.maximum(error, 0)

    error_norm = (error - error.min()) / (error.max() - error.min() + 1e-8)

    flat = error_norm.flatten()
    k = int(0.02 * len(flat))

    thresh_val = np.sort(flat)[-k]
    mask = (error_norm >= thresh_val).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest).astype(np.uint8)

    kernel = np.ones((11,11), np.uint8)
    mask = cv2.dilate(mask, kernel)

    return mask


def extract_anomaly_info(mask, frame_idx, frame_shape):

    h, w = frame_shape
    coords = np.column_stack(np.where(mask == 1))

    if len(coords) == 0:
        return None

    y_mean, x_mean = coords.mean(axis=0)

    region_x = "left" if x_mean < w/3 else "center" if x_mean < 2*w/3 else "right"
    region_y = "top" if y_mean < h/3 else "middle" if y_mean < 2*h/3 else "bottom"

    region = f"{region_y}-{region_x}"
    area = len(coords)

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    height = y_max - y_min
    width = x_max - x_min

    aspect_ratio = width / (height + 1e-5)
    compactness = area / ((width * height) + 1e-5)

    size_category = "medium" if area > 400 else "small" if area > 150 else "very small"

    return {
        "frame": int(frame_idx),
        "region": region,
        "area": int(area),
        "width": int(width),
        "height": int(height),
        "aspect_ratio": float(aspect_ratio),
        "compactness": float(compactness),
        "size_category": size_category
    }


# 🔥 Replace this with your Groq / OpenAI client
def generate_llm_explanation(info):

    if info is None:
        return "No anomaly detected."

    return f"A likely anomaly (bicycle-like object) is detected in the {info['region']} of the pedestrian walkway."


# =========================
# 🎬 VIDEO → SEQUENCE
# =========================

def video_to_sequence(video_file, window_size=16):

    cap = cv2.VideoCapture(video_file)
    frames = []

    while len(frames) < window_size:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (224,224))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = gray / 255.0

        frames.append(gray)

    cap.release()

    if len(frames) < window_size:
        return None

    frames = np.array(frames)[:, np.newaxis, :, :]
    frames = torch.tensor(frames, dtype=torch.float32)

    return frames.unsqueeze(0)  # (1, T, C, H, W)


# =========================
# 🔍 MAIN PIPELINE
# =========================

def analyze(video_path):

    seq = video_to_sequence(video_path)

    if seq is None:
        return None, None, None

    seq = seq.to(device)

    with torch.no_grad():
        recon = model(seq)

    error_seq = (seq - recon)**2
    error_seq = error_seq[0]  # remove batch

    frame_idx = 8

    mask = get_mask_from_error(error_seq, frame_idx)

    orig_frame = seq[0][frame_idx][0].cpu().numpy()
    orig_frame = (orig_frame * 255).astype(np.uint8)

    info = extract_anomaly_info(mask, frame_idx, orig_frame.shape)
    explanation = generate_llm_explanation(info)

    return orig_frame, mask, explanation


# =========================
# 🌐 STREAMLIT UI
# =========================
st.title("Video Anomaly Detection with 3D CNN + LLM Explanation")
st.video("assets/intro.mp4")
st.write("3D CNN + Explainable AI + LLM")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi"])

if uploaded_file:

    with open("temp.mp4", "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Analyzing..."):
        orig, mask, explanation = analyze("temp.mp4")

    if orig is not None:

        st.subheader("Original Frame")
        st.image(orig, clamp=True)

        st.subheader("Anomaly Mask")
        st.image(mask.astype(np.uint8) * 255)

        overlay = np.stack([orig]*3, axis=-1)
        overlay[mask == 1] = [255, 0, 0]

        st.subheader("Highlighted Anomaly")
        st.image(overlay)

        st.subheader("AI Explanation")
        st.write(explanation)

    else:
        st.error("Could not process video (too short)")