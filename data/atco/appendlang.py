import whisper
import os
import json
from tqdm import tqdm

model = whisper.load_model("medium")

META_FILES= ["metadata_fr_test.json","metadata_fr_train.json"]
DISK_DIR="/run/media/johnny/31c5407a-2da6-4ef8-95ec-d294c1afec38/"

for file in META_FILES:
    with open(file,'r') as f:
        data = json.load(f)
        for i in tqdm(range(len(data))):
            audio_path = os.path.join(DISK_DIR,data[i]["audio"])
            data[i]["lang"] = 'fr'
        f.close()
    
    with open(file,'w') as f:
        json.dump(data,f,indent=4)
        f.close()
        