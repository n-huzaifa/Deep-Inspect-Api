# Deep-Inspect-Api

Deep-Inspect-Api is a FastAPI application designed to analyze audio files to detect specific audio features. This API is particularly useful for tasks like distinguishing between genuine and deepfake audio.

## Features

- **Upload and Save Audio**: Upload audio files to the server for analysis.
- **Audio Analysis**: Extract features from audio files including MFCCs, LPCCs, VOT, burst time, and periodicity time.
- **Batch Processing**: Analyze multiple audio files concurrently.
- **Machine Learning Integration**: Utilize a pre-trained SVM classifier to predict and classify audio features.

## Getting Started

1. **Clone the repository**:

   ```sh
   git clone git@github.com:n-huzaifa/Deep-Inspect-Api.git
   cd deep-inspect-api
   ```

2. **Install dependencies**:

   ```sh
   pip install fastapi numpy joblib librosa praat-parselmouth
   ```

3. **Run the FastAPI server**:

   ```sh
   fastapi dev .\main.py
   ```

4. **Upload audio files**:
   - Use an API client like Postman or curl to send a POST request to `http://127.0.0.1:8000/upload` with the audio file.

## What the Code Does

- **Main Application**: `main.py` contains the FastAPI application, including an endpoint to upload audio files and save them to the server.
- **Inference and Analysis**: `inference.py` includes functions for extracting features from audio files and processing them with a pre-trained SVM classifier.

Ensure the paths to the SVM model and scaler are correct in the `inference.py` file. This setup allows you to analyze audio files for specific features and classify them using a machine learning model.

## License

This project is licensed under the [MIT License](https://opensource.org/license/mit), with the additional condition that any usage or distribution of this project must prominently display credit to the original author [n-huzaifa](https://github.com/n-huzaifa). Failure to provide proper credit constitutes a violation of the license terms.
