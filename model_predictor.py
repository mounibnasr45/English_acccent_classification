# accent_detector/model_predictor.py

import tensorflow as tf
import pickle
import numpy as np
from config import MODEL_PATH, LABEL_ENCODER_PATH, N_MFCC_EFFECTIVE, MAX_MFCC_FRAMES

# Suppress TensorFlow/Keras informational messages (optional, good for cleaner app output)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')


def load_accent_model():
    """Loads the pre-trained Keras model."""
    try:
        # When loading a model with custom objects (like custom layers or regularizers, if any),
        # you might need to provide a custom_objects dictionary.
        # For a standard Sequential model with common layers, this is usually not needed.
        model = tf.keras.models.load_model(MODEL_PATH)
        # Verify input shape
        expected_input_shape = (None, MAX_MFCC_FRAMES, N_MFCC_EFFECTIVE)
        if model.input_shape != expected_input_shape:
             print(f"Warning: Model input shape {model.input_shape} " \
                   f"differs from expected {expected_input_shape}. Ensure consistency.")
        return model
    except Exception as e:
        print(f"Error loading Keras model from {MODEL_PATH}: {e}")
        return None

def load_accent_label_encoder():
    """Loads the pickled LabelEncoder."""
    try:
        with open(LABEL_ENCODER_PATH, 'rb') as f:
            label_encoder = pickle.load(f)
        return label_encoder
    except Exception as e:
        print(f"Error loading LabelEncoder from {LABEL_ENCODER_PATH}: {e}")
        return None

def predict_accent(model, label_encoder, mfcc_features):
    """
    Predicts accent and confidence from MFCC features.
    Returns (predicted_accent_label, confidence_score_percentage).
    """
    if model is None or label_encoder is None or mfcc_features is None:
        return None, None

    try:
        # Reshape for model input (batch_size, num_frames, num_features)
        mfcc_reshaped = np.expand_dims(mfcc_features, axis=0)

        probabilities = model.predict(mfcc_reshaped)[0]  # Get probabilities for the single sample
        predicted_index = np.argmax(probabilities)
        
        # Check if predicted_index is within the range of known classes by the encoder
        if predicted_index < len(label_encoder.classes_):
            predicted_accent_label = label_encoder.inverse_transform([predicted_index])[0]
        else:
            print(f"Error: Predicted index {predicted_index} is out of bounds for label encoder classes (max: {len(label_encoder.classes_)-1}).")
            return "ErrorInPrediction", 0.0

        confidence_score = probabilities[predicted_index] * 100  # As percentage
        return predicted_accent_label, confidence_score
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None