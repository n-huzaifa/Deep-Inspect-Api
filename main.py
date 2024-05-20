import os

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from inference import analyze_audio_batch

app = FastAPI()

# Allow requests from your Chrome extension's origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["chrome-extension://jkapobccndghglakoceikpihckclaibd"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


UPLOAD_FOLDER = "uploads"  # Define the folder where uploaded files will be stored

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def save_file_to_server(file: UploadFile) -> str:
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    return file_path

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # Save the file to the server
    file_path = save_file_to_server(file)
    print("File saved to:", file_path)
    
    # Break down audio into 1-second clips and save
    
    # predictions, probabilities = analyze_audio_batch(['/'])
    # print("Predictions:", predictions)
    # print("Probabilities:", probabilities)
    # print_predictions(predictions, probabilities)

    
    return {"filename": file.filename, "file_path": file_path}