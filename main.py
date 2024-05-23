import os
import logging
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from inference import analyze_audio_batch, print_predictions, split_wav_file

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

def cleanup_files(file_paths):
    print(file_paths)
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logging.info(f"Deleted file: {file_path}")
            else:
                logging.warning(f"File not found: {file_path}")
        except Exception as e:
            logging.error(f"Error deleting file {file_path}: {e}")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Save the file to the server
        file_path = save_file_to_server(file)
        logging.info(f"File saved to: {file_path}")
        
        # Normalize the file path for Python
        normalized_file_path = os.path.normpath(file_path)
        logging.info(f"Normalized file path: {normalized_file_path}")
        
        # Break down audio into 1-second clips and save
        input_audio_paths = split_wav_file(normalized_file_path)
        logging.info(f"Created audio segments: {input_audio_paths}")
        
        cleanup_files(input_audio_paths)
        predictions, probabilities = analyze_audio_batch(input_audio_paths)
        logging.info(f"Predictions: {predictions}")
        logging.info(f"Probabilities: {probabilities}")

        # Clean up temporary files
        
        return {"filename": file.filename, "file_path": file_path}
    except Exception as e:
        logging.error(f"Error processing file: {e}")
        return {"error": str(e)}
