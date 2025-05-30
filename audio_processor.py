# accent_detector/audio_processor.py

import os
import librosa
import numpy as np
import yt_dlp # Make sure this is imported
import uuid
from config import (
    SAMPLE_RATE, DURATION, N_MFCC_BASE, HOP_LENGTH, MAX_MFCC_FRAMES,
    YDL_OPTS, TEMP_AUDIO_DIR, FFMPEG_BIN_DIRECTORY_PATH # <-- FFMPEG_BIN_DIRECTORY_PATH is imported
)
import traceback # <-- For detailed error logging

# Ensure librosa warnings are managed if necessary
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning, module='librosa')
# warnings.filterwarnings("ignore", category=FutureWarning, module='librosa')

if not os.path.exists(TEMP_AUDIO_DIR):
    os.makedirs(TEMP_AUDIO_DIR)

def download_audio_from_url(video_url):
    """
    Downloads audio from a video URL using yt-dlp.
    Saves to a temporary file and returns the path.
    """
    unique_filename_base = str(uuid.uuid4())
    output_template = os.path.join(TEMP_AUDIO_DIR, f"{unique_filename_base}.%(ext)s")

    # Use YDL_OPTS from config and add ffmpeg_location if available
    ydl_opts_dynamic = YDL_OPTS.copy()
    ydl_opts_dynamic['outtmpl'] = output_template

    # --- MODIFICATION START ---
    # Check if FFMPEG_BIN_DIRECTORY_PATH is set and is a valid directory path
    if FFMPEG_BIN_DIRECTORY_PATH and os.path.isdir(FFMPEG_BIN_DIRECTORY_PATH):
        ydl_opts_dynamic['ffmpeg_location'] = FFMPEG_BIN_DIRECTORY_PATH
    # --- MODIFICATION END ---

    # Print the options yt-dlp will use, for debugging
    # This existing debug block will now reflect if ffmpeg_location was set
    print(f"DEBUG: yt-dlp options being used: {ydl_opts_dynamic}")
    if 'ffmpeg_location' in ydl_opts_dynamic and ydl_opts_dynamic['ffmpeg_location']:
        print(f"DEBUG: ffmpeg_location is set to: {ydl_opts_dynamic['ffmpeg_location']}")
        if not os.path.isdir(ydl_opts_dynamic['ffmpeg_location']): # Should not happen if previous check passed
            print(f"DEBUG WARNING: The ffmpeg_location path '{ydl_opts_dynamic['ffmpeg_location']}' does not exist or is not a directory.")
        else:
            ffmpeg_exe = os.path.join(ydl_opts_dynamic['ffmpeg_location'], 'ffmpeg.exe')
            ffprobe_exe = os.path.join(ydl_opts_dynamic['ffmpeg_location'], 'ffprobe.exe')
            print(f"DEBUG: Checking for ffmpeg.exe: {ffmpeg_exe} - Exists: {os.path.exists(ffmpeg_exe)}")
            print(f"DEBUG: Checking for ffprobe.exe: {ffprobe_exe} - Exists: {os.path.exists(ffprobe_exe)}")
    else:
        print("DEBUG: ffmpeg_location is NOT explicitly set in ydl_opts_dynamic (either not in config, or path was invalid). yt-dlp will try to find it in PATH.")


    try:
        with yt_dlp.YoutubeDL(ydl_opts_dynamic) as ydl:
            print(f"DEBUG: Attempting to download and process: {video_url}")
            info_dict = ydl.extract_info(video_url, download=True)
            
            preferred_codec = ydl_opts_dynamic.get('postprocessors', [{}])[0].get('preferredcodec', 'mp3')
            downloaded_file_path = None
            for f_name in os.listdir(TEMP_AUDIO_DIR):
                if f_name.startswith(unique_filename_base) and f_name.endswith(f".{preferred_codec}"):
                    downloaded_file_path = os.path.join(TEMP_AUDIO_DIR, f_name)
                    break
            
            if downloaded_file_path and os.path.exists(downloaded_file_path):
                print(f"DEBUG: Successfully found downloaded file: {downloaded_file_path}")
                return downloaded_file_path
            else:
                print(f"DEBUG: Preferred codec file .{preferred_codec} not found directly. Searching for other files with base '{unique_filename_base}'...")
                for f_name in os.listdir(TEMP_AUDIO_DIR):
                    if f_name.startswith(unique_filename_base):
                        potential_path = os.path.join(TEMP_AUDIO_DIR, f_name)
                        print(f"DEBUG: Found potential fallback file: {potential_path}")
                print(f"Error: Postprocessed audio file with preferred codec .{preferred_codec} not found for base {unique_filename_base} in {TEMP_AUDIO_DIR}")
                return None

    except yt_dlp.utils.DownloadError as de: 
        print(f"CRITICAL YT-DLP DownloadError for URL '{video_url}': {de}")
        print("--- Full yt-dlp DownloadError Traceback ---")
        traceback.print_exc()
        print("-----------------------------------------")
        return None
    except Exception as e:
        print(f"CRITICAL Unexpected error downloading or processing audio from URL '{video_url}': {e}")
        print("--- Full Exception Traceback ---")
        traceback.print_exc()
        print("--------------------------------")
        return None

def load_and_prepare_audio(file_path):
    """
    Loads an audio file, resamples, and pads/truncates to a fixed duration.
    """
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

        target_raw_length = int(DURATION * SAMPLE_RATE)
        if len(y) < target_raw_length:
            y = np.pad(y, (0, target_raw_length - len(y)), 'constant')
        elif len(y) > target_raw_length:
            y = y[:target_raw_length]
        return y
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        traceback.print_exc() # Print traceback for loading errors too
        return None

def extract_mfcc_features(audio_segment):
    """
    Extracts MFCCs (including deltas and delta-deltas) and pads/truncates the sequence.
    """
    if audio_segment is None:
        return None
    try:
        mfccs = librosa.feature.mfcc(y=audio_segment, sr=SAMPLE_RATE, n_mfcc=N_MFCC_BASE, hop_length=HOP_LENGTH)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)

        combined_mfccs = np.concatenate((mfccs, delta_mfccs, delta2_mfccs), axis=0)
        mfccs_transposed = combined_mfccs.T

        if mfccs_transposed.shape[0] < MAX_MFCC_FRAMES:
            pad_width = MAX_MFCC_FRAMES - mfccs_transposed.shape[0]
            mfccs_padded = np.pad(mfccs_transposed, ((0, pad_width), (0, 0)), mode='constant')
        else:
            mfccs_padded = mfccs_transposed[:MAX_MFCC_FRAMES, :]
        return mfccs_padded
    except Exception as e:
        print(f"Error extracting MFCC features: {e}")
        traceback.print_exc() # Print traceback for MFCC errors
        return None

def cleanup_temp_file(file_path):
    """Removes a temporary file if it exists."""
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
            # print(f"Cleaned up temp file: {file_path}")
        except Exception as e:
            print(f"Error cleaning up temp file {file_path}: {e}")