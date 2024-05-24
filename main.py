from typing import List, Dict, Any
import os
import logging
from logging.handlers import RotatingFileHandler
from fastapi import Request
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from inference import analyze_audio_batch, print_predictions, split_wav_file, process_predictions_and_probabilities
import numpy as np
import json

app = FastAPI()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Add a handler for main.log
main_log_handler = RotatingFileHandler("main.log", maxBytes=10485760, backupCount=5)
main_log_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
main_log_handler.setFormatter(formatter)
logging.getLogger('').addHandler(main_log_handler)

FILE_METADATA_PATH = 'file_metadata.json'

# Allow requests from your Chrome extension's origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"  # Define the folder where uploaded files will be stored

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    logging.info(f"Created upload folder: {UPLOAD_FOLDER}")

def save_file_to_server(file: UploadFile) -> str:
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    logging.info(f"File saved to: {file_path}")
    return file_path

def cleanup_files(file_paths: List[str]):
    logging.info(file_paths)
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logging.info(f"Deleted file: {file_path}")
            else:
                logging.warning(f"File not found: {file_path}")
        except Exception as e:
            logging.error(f"Error deleting file {file_path}: {e}")

def load_file_metadata():
    if os.path.exists(FILE_METADATA_PATH):
        with open(FILE_METADATA_PATH, 'r') as f:
            return json.load(f)
    return []

def save_file_metadata(metadata):
    with open(FILE_METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        # Save the file to the server
        file_path = save_file_to_server(file)
        logging.info(f"File saved to: {file_path}")

        # Normalize the file path for Python
        normalized_file_path = os.path.normpath(file_path)
        logging.info(f"Normalized file path: {normalized_file_path}")

        # Load existing file metadata
        file_metadata = load_file_metadata()

        # Break down audio into 1-second clips and save
        input_audio_paths = split_wav_file(normalized_file_path)
        logging.info(f"Created audio segments: {input_audio_paths}")

        predictions, probabilities = analyze_audio_batch(input_audio_paths)
        logging.info(f"Predictions: {predictions}")
        logging.info(f"Probabilities: {probabilities}")
        logging.info(print_predictions(predictions, probabilities))
        process_predictions_and_probabilities(predictions, probabilities)

        # Clean up temporary files
        cleanup_files(input_audio_paths)

        # Convert numpy arrays to lists for serialization
        predictions = predictions.tolist()
        probabilities = probabilities.tolist()

        # Add new file metadata
        new_metadata = {
            "filename": file.filename,
            "file_path": normalized_file_path,
            "choice": "unchecked",
            "predictions": predictions,
            "probabilities": probabilities
        }
        file_metadata.append(new_metadata)

        # Save updated file metadata
        save_file_metadata(file_metadata)

        # Ensure predictions and probabilities can be serialized
        response = {
            "filename": file.filename,
            "file_path": file_path,
            "predictions": predictions,
            "probabilities": probabilities
        }
        return response
    except Exception as e:
        logging.error(f"Error processing file: {e}")
        return {"error": str(e)}
    

@app.post("/data-collect")
async def collect_data_feedback(request: Request):
    try:
        request_data = await request.json()
        filename = request_data.get("filename")
        file_path = request_data.get("filePath")
        choice = request_data.get("choice")

        # Load existing file metadata
        file_metadata = load_file_metadata()

        # Find the metadata for the given file
        file_metadata_entry = next((entry for entry in file_metadata if entry["filename"] == filename and entry["file_path"] == file_path), None)

        if file_metadata_entry:
            # Update the choice value
            file_metadata_entry["choice"] = choice

            # Save the updated file metadata
            save_file_metadata(file_metadata)

            return {"message": "Data feedback received successfully"}
        else:
            return {"error": "File metadata not found"}

    except Exception as e:
        logging.error(f"Error processing data feedback: {e}")
        return {"error": str(e)}