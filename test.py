#import os
import torch
import nemo.collections.asr as nemo_asr
#from io import BytesIO

def infer_greedy(files, asr_model):
    transcripts = asr_model.transcribe(files, batch_size=20)
    return transcripts


if __name__ == '__main__':
    model = "QuartzNet15x5_golos.nemo"
    asr_model = nemo_asr.models.EncDecCTCModel.restore_from(model)
    #asr_model.cuda()

    files = ["sound.wav"]
    #with open("sound.wav", 'rb') as f:
     #   data = f.read()

    hyps = infer_greedy(files, asr_model)
    print(hyps)