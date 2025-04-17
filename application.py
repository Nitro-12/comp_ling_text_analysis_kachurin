import numpy as np
import nemo.collections.asr as nemo_asr
from fastapi import File, UploadFile, FastAPI
from typing import List

if not hasattr(np, 'sctypes'):
    np.sctypes = {
        'int': [np.int8, np.int16, np.int32, np.int64],
        'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
        'float': [np.float16, np.float32, np.float64],
        'complex': [np.complex64, np.complex128],
    }

app = FastAPI()

# Загрузка модели
model_path = "QuartzNet15x5_golos_nemo.nemo"
asr_model = nemo_asr.models.EncDecCTCModel.restore_from(model_path)

# Очиcnrf конфигурацию от train/val/test секций
asr_model.cfg.train_ds = None
asr_model.cfg.validation_ds = None
asr_model.cfg.test_ds = None

# Сохранение загруженного файла на диск
def save_file(filename, data):
    with open(filename, 'wb') as f:
        f.write(data)

# Распознование аудиофайлов
def process_files(files):
    return asr_model.transcribe(files, batch_size=20)

# POST-запрос на загрузку файлов
@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    audioFiles = []
    for file in files:
        contents = await file.read()
        save_file(file.filename, contents)
        audioFiles.append(file.filename)
    result = process_files(audioFiles)
    return {"text": result}

# Проверочный корень
@app.get("/")
async def root():
    return {"message": "Server is working"}
