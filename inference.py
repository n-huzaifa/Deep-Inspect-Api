import os
import numpy as np
import joblib
import librosa
import parselmouth
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def extract_mfccs_(audio_data, sr, n_mfcc=20, n_fft=2048, hop_length=512):
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return np.mean(mfccs.T, axis=0)

def compute_lpcc(lpc_coeffs, sr):
    lpcc_coeffs = np.zeros_like(lpc_coeffs)
    lpcc_coeffs[0] = np.log(lpc_coeffs[0])
    i_sr = np.arange(1, len(lpc_coeffs)) / sr

    for i in range(1, len(lpc_coeffs)):
        lpcc_coeffs[i] = lpc_coeffs[i] + np.dot(lpcc_coeffs[1:i], lpc_coeffs[i-1:0:-1]) * i_sr[i-1]

    return lpcc_coeffs

def extract_lpccs_(audio_data, sr, order=19):
    lpc_coeffs = librosa.lpc(audio_data, order=order)
    lpcc_coeffs = compute_lpcc(lpc_coeffs, sr)
    return lpcc_coeffs.T

def estimate_vot(audio_path):
    sound = parselmouth.Sound(audio_path)
    pitch = sound.to_pitch()
    pitch_times = pitch.xs()
    vot_time = pitch_times[0] if len(pitch_times) > 0 else None
    return vot_time

def compute_energy_and_zcr(y, hop_length):
    energy = np.array([np.sum(y[i:i+hop_length]**2) for i in range(0, len(y), hop_length)])
    zcr = np.array([np.sum(np.diff(y[i:i+hop_length] > 0)) for i in range(0, len(y), hop_length)])
    return energy, zcr

def find_burst_and_periodicity(energy, zcr, hop_length, sr):
    energy_diff = np.diff(energy)
    burst_frame = np.argmax(energy_diff) + 1
    zcr_diff = np.diff(zcr)
    periodicity_frame = np.argmin(zcr_diff) + 1
    burst_time = burst_frame * hop_length / sr
    periodicity_time = periodicity_frame * hop_length / sr
    return burst_time, periodicity_time

def detect_burst_and_periodicity_onset(audio_data, sr, hop_length=512):
    energy, zcr = compute_energy_and_zcr(audio_data, hop_length)
    burst_time, periodicity_time = find_burst_and_periodicity(energy, zcr, hop_length, sr)
    return burst_time, periodicity_time

def extract_features(audio_path):
    try:
        audio_data, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        return None

    mfcc_features = extract_mfccs_(audio_data, sr)
    lpcc_features = extract_lpccs_(audio_data, sr)
    vot = estimate_vot(audio_path)
    burst_time, periodicity_time = detect_burst_and_periodicity_onset(audio_data, sr)
    vot_feature = np.array([vot, burst_time, periodicity_time])
    feature_vector = np.concatenate((mfcc_features, lpcc_features, vot_feature))
    return feature_vector

def process_audio(input_audio_path, scaler, svm_classifier):
    if not os.path.exists(input_audio_path):
        print(f"Error: The specified file {input_audio_path} does not exist.")
        return None, None
    elif not input_audio_path.lower().endswith(".wav"):
        print(f"Error: The specified file {input_audio_path} is not a .wav file.")
        return None, None

    features = extract_features(input_audio_path)

    if features is not None:
        features_scaled = scaler.transform(features.reshape(1, -1))
        prediction = svm_classifier.predict(features_scaled)
        probability = svm_classifier.predict_proba(features_scaled)
        return prediction[0], probability[0]
    else:
        print(f"Error: Unable to process the input audio {input_audio_path}.")
        return None, None

def analyze_audio_batch(input_audio_paths):
    start_time = time.time()

    model_filename = "/content/drive/MyDrive/FYP-Testing/svm_model_rbf_auto.pkl"
    scaler_filename = "/content/drive/MyDrive/FYP-Testing/scaler_rbf_auto.pkl"
    svm_classifier = joblib.load(model_filename)
    scaler = joblib.load(scaler_filename)

    predictions = []
    probabilities = []

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_audio, input_audio_path, scaler, svm_classifier) for input_audio_path in input_audio_paths]
        for future in as_completed(futures):
            prediction, probability = future.result()
            predictions.append(prediction)
            probabilities.append(probability)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total time taken: {elapsed_time:.2f} seconds")

    return np.array(predictions), np.array(probabilities)

def print_predictions(predictions, probabilities):
    print(f"{'Index':<6} || {'Class Predicted':<16} || {'Probability of Real':<20} || {'Probability of Fake':<20}")
    print("="*76)
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        class_predicted = "Genuine" if pred == 0 else "Deepfake"
        prob_real = prob[0]
        prob_fake = prob[1]
        print(f"{i:<6} || {class_predicted:<16} || {prob_real:<20.4f} || {prob_fake:<20.4f}")

# Export the function for reuse
__all__ = ['analyze_audio_batch','print_predictions']
