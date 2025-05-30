# accent_detector/app.py

import streamlit as st
import os # For cleanup
from config import APP_TITLE, URL_INPUT_LABEL, ANALYZE_BUTTON_TEXT
from audio_processor import (
    download_audio_from_url, load_and_prepare_audio,
    extract_mfcc_features, cleanup_temp_file
)
from model_predictor import (
    load_accent_model, load_accent_label_encoder, predict_accent
)

# --- Page Configuration (Optional) ---
st.set_page_config(page_title=APP_TITLE, layout="centered")

# --- Load Model and Encoder (Cached for performance) ---
@st.cache_resource # Use cache_resource for models, connections, etc.
def get_model():
    return load_accent_model()

@st.cache_resource
def get_label_encoder():
    return load_accent_label_encoder()

model = get_model()
label_encoder = get_label_encoder()

# --- Main Application UI ---
st.title(APP_TITLE)
st.write("This tool analyzes the English accent from a speaker in a public video.")
st.write(f"Currently trained to identify: {', '.join(label_encoder.classes_ if label_encoder else ['Loading...'])}")


video_url = st.text_input(URL_INPUT_LABEL, placeholder="https://www.youtube.com/watch?v=...")

if st.button(ANALYZE_BUTTON_TEXT):
    if not video_url:
        st.warning("Please enter a video URL.")
    elif model is None or label_encoder is None:
        st.error("Model or Label Encoder could not be loaded. Please check server logs.")
    else:
        temp_audio_file_path = None # To ensure cleanup
        try:
            with st.spinner("Downloading and extracting audio... This may take a moment."):
                temp_audio_file_path = download_audio_from_url(video_url)

            if temp_audio_file_path:
                st.success(f"Audio extracted: {os.path.basename(temp_audio_file_path)}")

                with st.spinner("Processing audio and extracting features..."):
                    raw_audio = load_and_prepare_audio(temp_audio_file_path)
                    if raw_audio is None:
                        st.error("Could not load or prepare the audio file.")
                    else:
                        mfcc_features = extract_mfcc_features(raw_audio)

                if mfcc_features is not None:
                    with st.spinner("Analyzing accent..."):
                        accent, confidence = predict_accent(model, label_encoder, mfcc_features)

                    if accent and confidence is not None:
                        st.subheader("Accent Analysis Results:")
                        st.metric(label="Predicted Accent", value=str(accent))
                        st.metric(label="Confidence Score", value=f"{confidence:.2f}%")

                        explanation = f"The speaker's accent is classified as **{accent}** " \
                                      f"with a confidence of **{confidence:.2f}%**. " \
                                      f"This model is trained primarily on {', '.join(label_encoder.classes_)} English accents."
                        if confidence < 60: # Arbitrary threshold for low confidence
                            explanation += " The confidence is somewhat low, which could indicate a different English " \
                                           "accent not in the training set, a non-native English speaker, " \
                                           "or challenges with audio quality/clarity."
                        st.info(explanation)
                    else:
                        st.error("Could not analyze the accent. Prediction failed.")
                else:
                    st.error("Could not extract features from the audio.")
            else:
                st.error("Failed to download or extract audio from the provided URL. "
                         "Please check the URL and ensure it's a publicly accessible video/audio.")

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
        finally:
            if temp_audio_file_path:
                cleanup_temp_file(temp_audio_file_path)