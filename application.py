import numpy as np
import nemo.collections.asr as nemo_asr
from fastapi import File, UploadFile, FastAPI
from typing import List
import os

if not hasattr(np, 'sctypes'):
    np.sctypes = {
        'int': [np.int8, np.int16, np.int32, np.int64],
        'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
        'float': [np.float16, np.float32, np.float64],
        'complex': [np.complex64, np.complex128],
    }

app = FastAPI()

model_path = "QuartzNet15x5_golos_nemo.nemo"
asr_model = nemo_asr.models.EncDecCTCModel.restore_from(model_path)

def save_file(filename, data):
    with open(filename, 'wb') as f:
        f.write(data)

def process_files(files):
    return asr_model.transcribe(files, batch_size=20)

@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    audioFiles = []
    for file in files:
        contents = await file.read()
        save_file(file.filename, contents)
        audioFiles.append(file.filename)
    return {"text": process_files(audioFiles)}

@app.get("/")
async def root():
    return {"message": "Server is working"}
