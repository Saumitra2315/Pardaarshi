import os
import pickle
import numpy as np
import sounddevice as sd
import soundfile as sf
from python_speech_features import mfcc

def extract_features(audio, sample_rate=16000, num_features=13):
    """
    Extract MFCC features from audio data.
    """
    try:
        # Extract MFCC features
        features = mfcc(audio, samplerate=sample_rate, numcep=num_features, nfft=2048)
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def recognize_voice():
    """
    Recognize the voice of the speaker by comparing with trained models.
    """
    # Configuration
    sample_rate = 16000
    duration = 5  # Seconds to record
    phrase = "Open Sesame"
    model_dir = "VoiceModels"
    min_score_threshold = -50  # Adjust based on empirical testing

    # Check if model directory exists
    if not os.path.exists(model_dir) or not os.listdir(model_dir):
        print("No trained voice models found! Train models first.")
        return

    # Capture audio
    print("Listening for the trigger phrase...")
    print(f"Please say '{phrase}'")
    recorded_audio = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()

    # Save audio for debugging purposes
    sf.write("captured_audio.wav", recorded_audio, sample_rate)

    # Extract MFCC features from the recorded audio
    recorded_audio = recorded_audio.flatten()
    mfcc_features = extract_features(recorded_audio, sample_rate)
    if mfcc_features is None:
        print("Failed to extract features from the recorded audio.")
        return

    print("Comparing with trained models...")
    best_score = float('-inf')
    best_user = None

    # Compare against all trained models
    for model_file in os.listdir(model_dir):
        if model_file.endswith("_model.pkl"):
            user_name = model_file.split("_model.pkl")[0]
            model_path = os.path.join(model_dir, model_file)

            # Load the GMM model
            with open(model_path, 'rb') as file:
                gmm = pickle.load(file)

            # Compute the log-likelihood score for all frames
            frame_scores = gmm.score_samples(mfcc_features)
            avg_score = np.mean(frame_scores)

            if avg_score > best_score:
                best_score = avg_score
                best_user = user_name

    # Decision based on threshold
    if best_score > min_score_threshold:
        print(f"Voice recognized! User: {best_user} (Score: {best_score:.2f})")
    else:
        print(f"Voice not recognized. Best match: {best_user} (Score: {best_score:.2f})")
