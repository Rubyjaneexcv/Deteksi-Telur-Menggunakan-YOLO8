import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Deteksi Telur Real-Time", page_icon="🥚", layout="wide")

# --- Judul Aplikasi ---
st.title("🥚 Aplikasi Deteksi Telur & Video")
st.write("Aplikasi ini menggunakan model YOLOv8 untuk mendeteksi telur asli dan objek mirip telur pada gambar dan video.")

# --- Sidebar ---
with st.sidebar:
    st.header("Informasi")
    st.write("Pilih salah satu tab untuk memulai deteksi.")
    st.write("1. **Upload Gambar**: Untuk file gambar statis.")
    st.write("2. **LIVE Webcam**: Untuk deteksi live stream.")
    st.write("3. **Upload Video**: Untuk file video.")
    st.write("---")
    st.write("Model yang digunakan adalah `best.pt` hasil pelatihan khusus.")

# --- Path ke Model ---
model_path = 'best.pt'

# --- Fungsi untuk Memuat Model (Dengan Perubahan) ---
@st.cache_resource
def load_yolo_model(path):
    """Memuat model YOLOv8 dari path."""
    # PENDEKATAN BARU: Kita tidak lagi menggunakan blok FIX.
    # Coba muat model dengan cara standar.
    # Jika ini masih gagal, ini menandakan masalah versi yang lebih dalam.
    try:
        model = YOLO(path)
        return model
    except Exception as e:
        # Menampilkan error yang lebih spesifik jika terjadi
        st.error(f"Gagal memuat model YOLO: {e}")
        st.error("Ini kemungkinan besar adalah masalah kompatibilitas versi antara PyTorch dan Ultralytics. Silakan coba Opsi Terakhir di bawah.")
        return None

# --- Memuat Model ---
model = load_yolo_model(model_path)
if model is None:
    st.stop() # Hentikan eksekusi jika model gagal dimuat

# --- Kelas untuk Pemrosesan Video Real-Time ---
class YOLOVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = self.model.predict(img)
        annotated_frame = results[0].plot()
        return annotated_frame

# --- TABS ---
tab_gambar, tab_video, tab_webcam_live = st.tabs(["🖼️ Upload Gambar", "📹 Upload Video", "LIVE 🎥 Webcam"])

# --- Logika untuk setiap tab (tidak ada perubahan di sini) ---
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

with tab_video:
    st.header("Deteksi Objek dari Video")
    uploaded_video = st.file_uploader("Pilih file video", type=["mp4", "mov", "avi"], key="uploader_video")
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        frame_placeholder = st.empty()
        st.info("Pemrosesan video dimulai...")
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                st.success("Pemrosesan video selesai!")
                break
            results = model.predict(frame)
            annotated_frame = results[0].plot()
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(annotated_frame_rgb, channels="RGB", use_column_width=True)
        cap.release()

with tab_webcam_live:
    st.header("Deteksi Objek dari Live Webcam")
    st.info("Tekan tombol 'START' di bawah untuk mengaktifkan webcam dan memulai deteksi real-time.")
    
    webrtc_streamer(
        key="live_detection",
        video_processor_factory=YOLOVideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )