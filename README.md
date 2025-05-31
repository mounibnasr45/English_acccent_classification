# AI-Powered English Accent Detector & Analyzer
DEMO :https://github.com/user-attachments/assets/2e9fba95-b288-4bfc-adff-d735f784a24e

## Project Overview

This project is an intelligent tool designed to analyze spoken English accents from public video URLs. It extracts audio, processes it, and employs a machine learning model to classify the speaker's accent ( American, British, Australian) and provide a confidence score for the detected English accent.

This tool was developed as a practical challenge for REM Waste, demonstrating the ability to build functional AI solutions for real-world problems, specifically to assist in evaluating the spoken English of potential candidates during the hiring process.

## Features

*   **URL-Based Audio Extraction:** Accepts public video URLs (YouTube, Loom, direct MP4 links) and automatically downloads and extracts the audio content.
*   **Accent Classification:** Identifies the primary English accent from a predefined set of classes (US, UK, Australian).
*   **Confidence Score:** Provides a percentage indicating the model's confidence in its accent classification and its assessment of the presence of a clear English accent.
*   **User-Friendly Interface:** A simple Streamlit web application for easy interaction and testing.
*   **Modular Design:** Code is structured into logical components for audio processing, model prediction, and UI.

## Technical Stack & Approach

This project leverages a Python-based stack:

*   **Core Language:** Python 3.x
*   **Web Framework:** Streamlit (for the interactive UI)
*   **Audio Processing:**
    *   `yt-dlp`: For robust downloading of audio from various video platforms.
    *   `librosa`: For audio loading, resampling, and feature extraction (MFCCs).
    *   `FFmpeg`: (Used by `yt-dlp` and `librosa` under the hood) For audio codec handling and conversion.
*   **Machine Learning:**
    *   `TensorFlow/Keras`: For building and loading the deep learning model (Convolutional Neural Network - CNN, or specify if different).
    *   `scikit-learn`: For the `LabelEncoder` used in training and prediction.
    *   `NumPy`: For numerical operations.
*   **Feature Engineering:** The model is trained on Mel-Frequency Cepstral Coefficients (MFCCs), including their first and second derivatives (deltas and delta-deltas), to capture timbral and dynamic characteristics of speech. Audio segments are standardized to a fixed duration (`5` seconds) and sampling rate (`16000` Hz) to match the training conditions.
*   **Model ArchitectureThe accent classification model is a Convolutional Neural Network (CNN) [or LSTM, or CNN-LSTM, etc.] trained on a dataset of diverse English accents. The architecture is designed to learn discriminative patterns from the MFCC features. *(If you used a pre-trained model as a base, mention it here and how you adapted it).*
*   **Deployment:** Streamlit Community Cloud

### Resourcefulness & Creative Choices:

*   **Leveraging `yt-dlp`:** Chosen for its versatility in handling a wide array of video URLs beyond just YouTube, and its robust error handling.
*   **Comprehensive MFCC Features:** Using MFCCs along with their deltas provides a richer representation of the audio signal, improving model accuracy.
*   **Standardized Input:** Fixing the audio duration and sample rate ensures consistency between training and inference, leading to more reliable predictions.
*   **Focus on Practicality:** The Streamlit UI makes the tool immediately usable and testable, aligning with the challenge's emphasis on a working solution.

## How It Works (Pipeline)

1.  **Input URL:** The user provides a public video URL.
2.  **Audio Download & Extraction:** `yt-dlp` downloads the best quality audio and converts it to a temporary MP3 (or WAV) file.
3.  **Audio Loading & Preprocessing:** `librosa` loads the audio, resamples it to `16000 Hz`, converts it to mono, and truncates/pads it to a fixed `5-second` duration.
4.  **Feature Extraction:** MFCCs (base, delta, delta-delta) are extracted from the audio segment, resulting in a feature matrix of shape `(MAX_MFCC_FRAMES, N_MFCC_EFFECTIVE)`.
5.  **Model Prediction:** The pre-trained Keras model (`accent_model.h5`) takes the MFCC features as input.
6.  **Output Generation:**
    *   The model outputs probabilities for each trained accent class.
    *   The `LabelEncoder` (`label_encoder.pkl`) maps the highest probability class index back to its accent label ("US", "UK","AUS").
    *   The confidence score is the probability of the predicted class.
    *   A brief summary/explanation is presented to the user.
7.  **Cleanup:** The temporary audio file is deleted.

## Getting Started (Running Locally)

To run this application on your local machine:

1.  **Clone the repository:**
    ```bash
    git clone [Link to your GitHub Repository]
    cd [repository-name]
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    Ensure you have FFmpeg installed and accessible in your system's PATH. If not, download it from [ffmpeg.org](https://ffmpeg.org/download.html) and add its `bin` directory to your PATH, or update `FFMPEG_BIN_DIRECTORY_PATH` in `accent_detector/config.py`.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit app:**
    Navigate to the directory containing `app.py` (if your `app.py` is in the root, you are already there after `cd [repository-name]`).
    ```bash
    streamlit run app.py
    ```
    The application should open in your web browser.

##Model Information & Training

*   **Trained Accents:** The model is currently trained to recognize the following English accents:
    *   `US`
    *   `UK`
    *   `Australia`
*   **Dataset:** *(Briefly mention the type of dataset used if you trained it, e.g., "The model was trained on a curated dataset comprising audio clips from [source like Common Voice, VoxForge, YouTube educational channels] representing diverse speakers for each accent class." If you used a pre-existing model, state that).*
*   **Evaluation:** *(If you trained it, briefly mention how it was evaluated, e.g., "The model achieved an accuracy of X% on a held-out test set.").*

## Limitations & Future Work

*   **Accent Coverage:** The current model is limited to the accents it was trained on (US,Uk, AUS). Performance on other English accents or non-native speakers may vary.
*   **Audio Quality Dependency:** Prediction accuracy is sensitive to audio quality. Clear speech with minimal background noise yields the best results.
*   **Fixed Duration Analysis:** The system analyzes only the first `5` seconds of audio. This might not capture enough information for very nuanced accents or varied speech.
*   **Single Speaker Assumption:** The tool assumes a single primary speaker in the analyzed segment.
*   **Confidence Nuance:** A high confidence score indicates the model is confident in its *classification among the known classes*, not necessarily absolute proof of that accent if the true accent is outside the training set.
