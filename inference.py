import os
import numpy as np
import joblib
import librosa
import soundfile as sf
import parselmouth
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Configure logging
logging.basicConfig(filename='main.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def split_wav_file(path):
    try:
        logging.info(f"Loading audio file: {path}")
        y, sr = librosa.load(path, sr=None)
        sample_length = sr

        num_segments = len(y) // sample_length
        segmented_audio = []

        for i in range(num_segments):
            start = i * sample_length
            end = (i + 1) * sample_length
            segment = y[start:end]
            temp_path = f"segment_{i}.wav"
            sf.write(temp_path, segment, sr)
            segmented_audio.append(temp_path)
            logging.info(f"Segmented file saved: {temp_path}")

        return segmented_audio
    except Exception as e:
        logging.error(f"Error in split_wav_file: {e}")
        return []

def extract_mfccs_(audio_data, sr, n_mfcc=20, n_fft=2048, hop_length=512):
    try:
        logging.info("Extracting MFCCs")
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        logging.error(f"Error in extract_mfccs_: {e}")
        return np.array([])

def compute_lpcc(lpc_coeffs, sr):
    try:
        logging.info("Computing LPCC")
        lpcc_coeffs = np.zeros_like(lpc_coeffs)
        lpcc_coeffs[0] = np.log(lpc_coeffs[0])
        i_sr = np.arange(1, len(lpc_coeffs)) / sr

        for i in range(1, len(lpc_coeffs)):
            lpcc_coeffs[i] = lpc_coeffs[i] + np.dot(lpcc_coeffs[1:i], lpc_coeffs[i-1:0:-1]) * i_sr[i-1]

        return lpcc_coeffs
    except Exception as e:
        logging.error(f"Error in compute_lpcc: {e}")
        return np.array([])

def extract_lpccs_(audio_data, sr, order=19):
    try:
        logging.info("Extracting LPCCs")
        lpc_coeffs = librosa.lpc(audio_data, order=order)
        lpcc_coeffs = compute_lpcc(lpc_coeffs, sr)
        return lpcc_coeffs.T
    except Exception as e:
        logging.error(f"Error in extract_lpccs_: {e}")
        return np.array([])

def estimate_vot(audio_path):
    try:
        logging.info(f"Estimating VOT for: {audio_path}")
        sound = parselmouth.Sound(audio_path)
        pitch = sound.to_pitch()
        pitch_times = pitch.xs()
        vot_time = pitch_times[0] if len(pitch_times) > 0 else None
        return vot_time
    except Exception as e:
        logging.error(f"Error in estimate_vot: {e}")
        return None

def compute_energy_and_zcr(y, hop_length):
    try:
        logging.info("Computing energy and ZCR")
        energy = np.array([np.sum(y[i:i+hop_length]**2) for i in range(0, len(y), hop_length)])
        zcr = np.array([np.sum(np.diff(y[i:i+hop_length] > 0)) for i in range(0, len(y), hop_length)])
        return energy, zcr
    except Exception as e:
        logging.error(f"Error in compute_energy_and_zcr: {e}")
        return np.array([]), np.array([])

def find_burst_and_periodicity(energy, zcr, hop_length, sr):
    try:
        logging.info("Finding burst and periodicity")
        energy_diff = np.diff(energy)
        burst_frame = np.argmax(energy_diff) + 1
        zcr_diff = np.diff(zcr)
        periodicity_frame = np.argmin(zcr_diff) + 1
        burst_time = burst_frame * hop_length / sr
        periodicity_time = periodicity_frame * hop_length / sr
        return burst_time, periodicity_time
    except Exception as e:
        logging.error(f"Error in find_burst_and_periodicity: {e}")
        return None, None

def detect_burst_and_periodicity_onset(audio_data, sr, hop_length=512):
    try:
        logging.info("Detecting burst and periodicity onset")
        energy, zcr = compute_energy_and_zcr(audio_data, hop_length)
        burst_time, periodicity_time = find_burst_and_periodicity(energy, zcr, hop_length, sr)
        return burst_time, periodicity_time
    except Exception as e:
        logging.error(f"Error in detect_burst_and_periodicity_onset: {e}")
        return None, None

def extract_features(audio_path):
    try:
        logging.info(f"Extracting features from: {audio_path}")
        audio_data, sr = librosa.load(audio_path, sr=None)

        mfcc_features = extract_mfccs_(audio_data, sr)
        lpcc_features = extract_lpccs_(audio_data, sr)
        vot = estimate_vot(audio_path)
        burst_time, periodicity_time = detect_burst_and_periodicity_onset(audio_data, sr)
        vot_feature = np.array([vot, burst_time, periodicity_time])
        feature_vector = np.concatenate((mfcc_features, lpcc_features, vot_feature))
        return feature_vector
    except Exception as e:
        logging.error(f"Error in extract_features: {e}")
        return None

def process_audio(input_audio_path, scaler, svm_classifier):
    try:
        logging.info(f"Processing audio: {input_audio_path}")

        if not os.path.exists(input_audio_path):
            logging.error(f"File does not exist: {input_audio_path}")
            return None, None
        elif not input_audio_path.lower().endswith(".wav"):
            logging.error(f"Not a .wav file: {input_audio_path}")
            return None, None

        features = extract_features(input_audio_path)

        if features is not None:
            features_scaled = scaler.transform(features.reshape(1, -1))
            prediction = svm_classifier.predict(features_scaled)
            probability = svm_classifier.predict_proba(features_scaled)
            return prediction[0], probability[0]
        else:
            logging.error(f"Unable to process input audio: {input_audio_path}")
            return None, None
    except Exception as e:
        logging.error(f"Error in process_audio: {e}")
        return None, None

def analyze_audio_batch(input_audio_paths):
    start_time = time.time()
    logging.info("Starting audio batch analysis")

    try:
        model_filename = "./svm_model_142_rbf.pkl"
        scaler_filename = "./scaler_142_rbf.pkl"
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
        logging.info(f"Total time taken: {elapsed_time:.2f} seconds")

        return np.array(predictions), np.array(probabilities)
    except Exception as e:
        logging.error(f"Error in analyze_audio_batch: {e}")
        return np.array([]), np.array([])

def print_predictions(predictions, probabilities):
    try:
        logging.info("Printing predictions and probabilities")
        print(f"{'Index':<6} || {'Class Predicted':<16} || {'Probability of Real':<20} || {'Probability of Fake':<20}")
        print("="*76)
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            class_predicted = "Genuine" if pred == 0 else "Deepfake"
            prob_real = prob[0]
            prob_fake = prob[1]
            print(f"{i:<6} || {class_predicted:<16} || {prob_real:<20.4f} || {prob_fake:<20.4f}")
    except Exception as e:
        logging.error(f"Error in print_predictions: {e}")

def process_predictions_and_probabilities(predictions, probabilities):
    try:
        logging.info("Processing predictions and probabilities")
        count_0 = np.count_nonzero(predictions == 0)
        count_1 = np.count_nonzero(predictions == 1)

        avg_genuine_prob = np.mean(probabilities[:, 0])
        avg_deepfake_prob = np.mean(probabilities[:, 1])

        return [count_0, count_1, avg_genuine_prob, avg_deepfake_prob]
    except Exception as e:
        logging.error(f"Error in process_predictions_and_probabilities: {e}")
        return []

__all__ = ['analyze_audio_batch', 'print_predictions', 'split_wav_file','process_predictions_and_probabilities']
