import os
import pickle
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.io.wavfile import read
from python_speech_features import mfcc

def extract_features(audio_path, sample_rate=16000, num_features=13):
    """
    Extract MFCC features from an audio file.
    """
    try:
        # Read the audio file
        sample_rate, audio = read(audio_path)
        # Extract MFCC features
        features = mfcc(audio, samplerate=sample_rate, numcep=num_features, nfft=2048)
        return features
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def train_voice_model():
    """
    Train a voice model using audio files in the 'TrainingVoice' directory.
    Saves a GMM model for each user.
    """
    training_dir = "TrainingVoice"
    model_dir = "VoiceModels"

    # Ensure the model directory exists
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Group audio files by user
    user_files = {}
    for file_name in os.listdir(training_dir):
        if file_name.endswith(".wav"):
            user_name = file_name.split("_")[0]
            if user_name not in user_files:
                user_files[user_name] = []
            user_files[user_name].append(os.path.join(training_dir, file_name))

    print("Starting voice model training...")
    
    for user, files in user_files.items():
        print(f"Training model for user: {user}")
        features = np.array([])

        for file_path in files:
            mfcc_features = extract_features(file_path)
            if mfcc_features is not None:
                if features.size == 0:
                    features = mfcc_features
                else:
                    features = np.vstack((features, mfcc_features))

        if features.size == 0:
            print(f"No valid features found for user {user}. Skipping.")
            continue

        # Train a Gaussian Mixture Model (GMM) for the user
        gmm = GaussianMixture(n_components=16, covariance_type='diag', max_iter=200, random_state=42)
        gmm.fit(features)

        # Save the trained model
        model_path = os.path.join(model_dir, f"{user}_model.pkl")
        with open(model_path, 'wb') as model_file:
            pickle.dump(gmm, model_file)

        print(f"Model for {user} saved at {model_path}")

    print("Training complete.")

