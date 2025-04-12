# Фикс для NumPy
import numpy as np
import torch
import os
import nemo.collections.asr as nemo_asr

def infer_greedy(files, asr_model):
    # Проверка файлов перед обработкой
    for file in files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Audio file {file} not found!")
    
    transcripts = asr_model.transcribe(files, batch_size=20)
    return transcripts

if __name__ == '__main__':
    model_path = "QuartzNet15x5_golos_nemo.nemo"
    
    # Проверка существования модели
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found!")
        exit()
    
    # Загрузка модели
    asr_model = nemo_asr.models.EncDecCTCModel.restore_from(model_path)
    
    # Укажите абсолютный путь к файлу
    audio_file = "/comp_ling/lab1_kachurin/001ce26c07c20eaa0d666b824c6c6924.wav"
    
    # Проверка формата файла
    if not audio_file.endswith('.wav'):
        print("Only WAV files are supported!")
        exit()
    
    files = [audio_file]
    
    try:
        hyps = infer_greedy(files, asr_model)
        print(hyps)
    except Exception as e:
        print(f"Error during inference: {str(e)}")
