# accent_detector/config.py

import os
import librosa

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "accent_model.h5")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")
TEMP_AUDIO_DIR = os.path.join(BASE_DIR, "temp_audio_files") # For downloaded audio
FFMPEG_BIN_DIRECTORY_PATH=r"C:\Users\mouni\Downloads\ffmpeg-2025-05-29-git-75960ac270-essentials_build\ffmpeg-2025-05-29-git-75960ac270-essentials_build\bin"
# --- Audio Processing Parameters ---
SAMPLE_RATE = 16000  # Must match the rate used for training
DURATION = 5         # Duration in seconds, must match training
N_MFCC_BASE = 20     # Base MFCCs, must match training
HOP_LENGTH = 512     # Must match training
# N_MFCC_EFFECTIVE will be N_MFCC_BASE * 3 (MFCC, Delta, Delta-Delta)
N_MFCC_EFFECTIVE = N_MFCC_BASE * 3
MAX_MFCC_FRAMES = int(librosa.time_to_frames(DURATION, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)) + 1 # Must match training

# --- Model & Prediction ---
# The classes your model was trained on (from label_encoder.classes_)
# This is just for reference or if you need to display them,
# the label_encoder.pkl handles the actual mapping.
# EXPECTED_CLASSES = ["US", "UK", "Australia"] # Or whatever your classes are

# --- yt-dlp Options ---
YDL_OPTS = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3', # wav is also good, mp3 is smaller
        'preferredquality': '192',
    }],
    'quiet': True,
    'no_warnings': True,
    # 'outtmpl': defined dynamically in audio_processor
}

# --- UI Texts (Optional, but good for consistency) ---
APP_TITLE = "English Accent Detector"
URL_INPUT_LABEL = "Enter public video URL (e.g., Loom, YouTube, MP4 link):"
ANALYZE_BUTTON_TEXT = "Analyze Accent"