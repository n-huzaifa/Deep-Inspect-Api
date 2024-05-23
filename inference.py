import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import joblib
import librosa
import numpy as np
import parselmouth
import soundfile as sf

# Set up logging
log_filename = 'audio_processing.log'
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to split a WAV file into segments
def split_wav_file(path):
    logging.info(f"Splitting the WAV file: {path}")
    try:
        y, sr = librosa.load(path, sr=None)
        logging.info(f"Loaded audio data with sample rate: {sr}")
        sample_length = sr
        num_segments = len(y) // sample_length
        segmented_audio = []

        for i in range(num_segments):
            start = i * sample_length
            end = (i + 1) * sample_length
            segment = y[start:end]
            temp_path = f"segment_{i}.wav"
            sf.write(temp_path, segment, sr)
            logging.info(f"Created segment: {temp_path}")
            segmented_audio.append(temp_path)
        return segmented_audio
    except Exception as e:
        logging.error(f"Error splitting WAV file {path}: {e}")
        return []

# Function to extract Mel-frequency cepstral coefficients (MFCCs) from audio data
def extract_mfccs_(audio_data, sr, n_mfcc=20, n_fft=2048, hop_length=512):
    try:
        logging.info(f"Extracting MFCCs with n_mfcc={n_mfcc}, n_fft={n_fft}, hop_length={hop_length}")
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        logging.info(f"MFCC shape: {mfccs.shape}")
        mfcc_features = np.mean(mfccs.T, axis=0)
        logging.info(f"MFCC features shape: {mfcc_features.shape}")
        return mfcc_features
    except Exception as e:
        logging.error(f"Error extracting MFCCs: {e}")
        return np.array([])

# Function to compute Linear Prediction Cepstral Coefficients (LPCCs)
def compute_lpcc(lpc_coeffs, sr):
    lpcc_coeffs = np.zeros_like(lpc_coeffs)
    lpcc_coeffs[0] = np.log(lpc_coeffs[0])
    i_sr = np.arange(1, len(lpc_coeffs)) / sr

    for i in range(1, len(lpc_coeffs)):
        lpcc_coeffs[i] = lpc_coeffs[i] + np.dot(lpcc_coeffs[1:i], lpc_coeffs[i-1:0:-1]) * i_sr[i-1]

    return lpcc_coeffs

# Function to extract LPCCs from audio data
def extract_lpccs_(audio_data, sr, order=19):
    try:
        logging.info(f"Extracting LPCCs with order={order}")
        lpc_coeffs = librosa.lpc(audio_data, order=order)
        logging.info(f"LPC coefficients shape: {lpc_coeffs.shape}")
        lpcc_coeffs = compute_lpcc(lpc_coeffs, sr)
        logging.info(f"LPCC coefficients shape: {lpcc_coeffs.shape}")
        return lpcc_coeffs.T
    except Exception as e:
        logging.error(f"Error extracting LPCCs: {e}")
        return np.array([])

# Function to estimate Voice Onset Time (VOT) from audio
def estimate_vot(audio_path):
    try:
        logging.info(f"Estimating Voice Onset Time (VOT) for {audio_path}")
        sound = parselmouth.Sound(audio_path)
        pitch = sound.to_pitch()
        pitch_times = pitch.xs()
        vot_time = pitch_times[0] if len(pitch_times) > 0 else None
        logging.info(f"Voice Onset Time: {vot_time}")
        return vot_time
    except Exception as e:
        logging.error(f"Error estimating VOT: {e}")
        return None

# Function to compute energy and zero crossing rate (ZCR) from audio data
def compute_energy_and_zcr(y, hop_length):
    try:
        logging.info(f"Computing energy and zero crossing rate (ZCR) with hop_length={hop_length}")
        energy = np.array([np.sum(y[i:i+hop_length]**2) for i in range(0, len(y), hop_length)])
        zcr = np.array([np.sum(np.diff(y[i:i+hop_length] > 0)) for i in range(0, len(y), hop_length)])
        logging.info(f"Energy shape: {energy.shape}, ZCR shape: {zcr.shape}")
        return energy, zcr
    except Exception as e:
        logging.error(f"Error computing energy and ZCR: {e}")
        return np.array([]), np.array([])

# Function to find burst and periodicity onset from energy and ZCR
def find_burst_and_periodicity(energy, zcr, hop_length, sr):
    try:
        logging.info("Finding burst and periodicity onset")
        energy_diff = np.diff(energy)
        burst_frame = np.argmax(energy_diff) + 1
        zcr_diff = np.diff(zcr)
        periodicity_frame = np.argmin(zcr_diff) + 1
        burst_time = burst_frame * hop_length / sr
        periodicity_time = periodicity_frame * hop_length / sr
        logging.info(f"Burst time: {burst_time}, Periodicity time: {periodicity_time}")
        return burst_time, periodicity_time
    except Exception as e:
        logging.error(f"Error finding burst and periodicity: {e}")
        return None, None

# Function to detect burst and periodicity onset from audio data
def detect_burst_and_periodicity_onset(audio_data, sr, hop_length=512):
    logging.info("Detecting burst and periodicity onset")
    try:
        energy, zcr = compute_energy_and_zcr(audio_data, hop_length)
        burst_time, periodicity_time = find_burst_and_periodicity(energy, zcr, hop_length, sr)
        return burst_time, periodicity_time
    except Exception as e:
        logging.error(f"Error detecting burst and periodicity onset: {e}")
        return None, None

def extract_features(audio_path):
    logging.info(f"Extracting features for {audio_path}")
    try:
        audio_data, sr = librosa.load(audio_path, sr=None)
        logging.info(f"Loaded audio data with sample rate: {sr}")
    except Exception as e:
        logging.error(f"Error loading audio file {audio_path}: {e}")
        return None

    mfcc_features = extract_mfccs_(audio_data, sr, n_mfcc=20)
    lpcc_features = extract_lpccs_(audio_data, sr, order=19)  # LPCCs with order 19 to match the expected 20 features
    vot = estimate_vot(audio_path) or 0
    burst_time, periodicity_time = detect_burst_and_periodicity_onset(audio_data, sr)
    vot_feature = np.array([vot, burst_time, periodicity_time])

    feature_vector = np.concatenate((mfcc_features, lpcc_features, vot_feature))
    logging.info(f"Feature vector shape: {feature_vector.shape}")
    assert feature_vector.shape[0] == 43, f"Expected 43 features, but got {feature_vector.shape[0]}"

    return feature_vector

# Function to process audio data and make predictions
def process_audio(input_audio_path, scaler, svm_classifier):
    logging.info(f"Processing audio: {input_audio_path}")
    try:
        if not os.path.exists(input_audio_path):
            logging.error(f"Error: The specified file {input_audio_path} does not exist.")
            return None, None
        elif not input_audio_path.lower().endswith(".wav"):
            logging.error(f"Error: The specified file {input_audio_path} is not a .wav file.")
            return None, None

        features = extract_features(input_audio_path)

        if features is not None:
            features_scaled = scaler.transform(features.reshape(1, -1))
            prediction = svm_classifier.predict(features_scaled)
            probability = svm_classifier.predict_proba(features_scaled)
            return prediction[0], probability[0]
        else:
            logging.error(f"Error: Unable to process the input audio {input_audio_path}.")
            return None, None
    except Exception as e:
        logging.error(f"Error processing audio {input_audio_path}: {e}")
        return None, None

# Function to analyze a batch of audio files
def analyze_audio_batch(input_audio_paths):
    start_time = time.time()
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
    logging.info(f"Total time taken: {elapsed_time:.2f} seconds")

    return np.array(predictions), np.array(probabilities)

# Function to print predictions
def print_predictions(predictions, probabilities):
    logging.info(f"{'Index':<6} || {'Class Predicted':<16} || {'Probability of Real':<20} || {'Probability of Fake':<20}")
    logging.info("="*76)
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        class_predicted = "Genuine" if pred == 0 else "Deepfake"
        prob_real = prob[0]
        prob_fake = prob[1]
        logging.info(f"{i:<6} || {class_predicted:<16} || {prob_real:<20.4f} || {prob_fake:<20.4f}")

model_filename = "./svm_model_142_rbf.pkl"
scaler_filename = "./scaler_142_rbf.pkl"
svm_classifier = joblib.load(model_filename)
scaler = joblib.load(scaler_filename)

# Export the function for reuse
__all__ = ['analyze_audio_batch', 'print_predictions', 'split_wav_file']
