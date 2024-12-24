import whisper
import os
import json
from tqdm import tqdm

model = whisper.load_model("medium")

META_FILES= ["metadata_train.json"]
DISK_DIR="/run/media/johnny/31c5407a-2da6-4ef8-95ec-d294c1afec38/"

def recog_lang(audio_path) -> str:
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    return max(probs, key=probs.get)
    
for file in META_FILES:
    with open(file,'r') as f:
        data = json.load(f)
        for i in tqdm(range(len(data))):
            audio_path = os.path.join(DISK_DIR,data[i]["audio"])
            data[i]["lang"] = recog_lang(audio_path)
        f.close()
    
    with open(file,'w') as f:
        json.dump(data,f,indent=4)
        f.close()
        