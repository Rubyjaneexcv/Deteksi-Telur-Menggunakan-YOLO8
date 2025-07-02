import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ============================================
# FIX UNTUK ERROR PYTORCH UNSUPPORTED GLOBAL
# ============================================
import torch.serialization
from ultralytics.nn.tasks import DetectionModel, SegmentationModel, PoseModel, ClassificationModel, OBBModel
torch.serialization.add_safe_globals([DetectionModel, SegmentationModel, PoseModel, ClassificationModel, OBBModel])
# ============================================
# AKHIR DARI FIX
# ============================================


# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Deteksi Telur Real-Time", page_icon="🥚", layout="wide")

# Judul aplikasi
st.title("🥚 Aplikasi Deteksi Telur & Video")
st.write("Aplikasi ini menggunakan model YOLOv8 untuk mendeteksi telur asli dan objek mirip telur pada gambar dan video.")

# Sidebar untuk informasi
with st.sidebar:
    st.header("Informasi")
    st.write("Pilih salah satu tab untuk memulai deteksi.")
    st.write("1. **Upload Gambar**: Untuk file gambar statis.")
    st.write("2. **Webcam**: Untuk mengambil foto langsung.")
    st.write("3. **Upload Video**: Untuk file video.")
    st.write("---")
    st.write("Model yang digunakan adalah `best.pt` hasil pelatihan khusus.")
    
# Path ke file model. Pastikan 'best.pt' ada di folder yang sama.
model_path = 'best.pt'

# Fungsi untuk memuat model dengan cache agar lebih cepat
@st.cache_resource
def load_yolo_model(path):
    """Memuat model YOLOv8 dari path."""
    return YOLO(path)

# Memuat model
try:
    model = load_yolo_model(model_path)
except Exception as e:
    st.error(f"Error memuat model: {e}")
    st.stop()

# Membuat tab untuk pilihan input
tab_gambar, tab_webcam, tab_video = st.tabs(["🖼️ Upload Gambar", "📷 Ambil Gambar dari Webcam", "📹 Upload Video"])

# --- TAB UPLOAD GAMBAR ---
with tab_gambar:
    st.header("Deteksi Objek dari Gambar")
    uploaded_file = st.file_uploader("Pilih file gambar", type=["jpg", "jpeg", "png"], key="uploader_gambar")
    if uploaded_file:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Gambar Asli", use_column_width=True)
        results = model.predict(image)
        annotated_image = results[0].plot()
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        with col2:
            st.image(annotated_image_rgb, caption="Hasil Deteksi", use_column_width=True)

# --- TAB WEBCAM ---
with tab_webcam:
    st.header("Deteksi Objek dari Webcam")
    camera_picture = st.camera_input("Arahkan webcam ke objek dan klik 'Take photo'")
    if camera_picture:
        image = Image.open(camera_picture)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Gambar dari Webcam", use_column_width=True)
        results = model.predict(image)
        annotated_image = results[0].plot()
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        with col2:
            st.image(annotated_image_rgb, caption="Hasil Deteksi", use_column_width=True)

# --- TAB UPLOAD VIDEO ---
with tab_video:
    st.header("Deteksi Objek dari Video")
    uploaded_video = st.file_uploader("Pilih file video", type=["mp4", "mov", "avi"], key="uploader_video")
    
    if uploaded_video is not None:
        # Buat file sementara untuk menyimpan video yang diupload
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        
        # Buka video menggunakan OpenCV
        cap = cv2.VideoCapture(tfile.name)
        
        # Buat placeholder untuk menampilkan video frame demi frame
        frame_placeholder = st.empty()
        
        st.info("Pemrosesan video dimulai... Tekan 'Stop' di pojok kanan atas untuk berhenti.")
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                st.success("Pemrosesan video selesai!")
                break
            
            # Lakukan deteksi pada setiap frame
            results = model.predict(frame)
            annotated_frame = results[0].plot()
            
            # Tampilkan frame yang sudah dianotasi
            # Konversi warna dari BGR (OpenCV) ke RGB (Streamlit)
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(annotated_frame_rgb, channels="RGB", use_column_width=True)
        
        # Lepaskan resource setelah selesai
        cap.release()