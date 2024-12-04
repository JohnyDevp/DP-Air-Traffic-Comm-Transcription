# Description: Synthesize the dataset to format used as nn input

import os, json
from scipy.io import wavfile
import numpy as np
from datasets import Dataset, DatasetDict, Audio
from tqdm import tqdm
import time

def normalize_audio(audio_array): # necessary to normalize the audio (datasets expect values between -1 and 1)
    return audio_array / np.max(np.abs(audio_array))

# function overtaken from whisper finetuning script
def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids **** CHANGED FROM **sentence** TO **transcription**
    batch["labels_fullts"] = tokenizer(batch["full_ts"]).input_ids
    batch["labels_shortts"] = tokenizer(batch["short_ts"]).input_ids
    return batch


if __name__ == "__main__":
    # must be set properly according to the wav paths stored in the metafile,
    # because the paths and recording dir will be combined
    
    # thats set for apimod
    # RECORDINGS_DIR="apimod/01_01_EL_LN_UJ_VV_YADA" 
    # META_FILE="apimod/01_01_EL_LN_UJ_VV_YADA/metadata.json"

    RECORDINGS_DIR="/run/media/johnny/31c5407a-2da6-4ef8-95ec-d294c1afec38/" 
    META_FILE="malorca/metadata_dev2.json"
    SAVE_DIR="malorca"

    # load the metadata
    meta_data = json.load(open(META_FILE, mode='r'))

    # load the wav files to the metadata
    for i, rec in tqdm(enumerate(meta_data)):
        # apimod
        # sampling_rate, wav_data = wavfile.read(os.path.join(RECORDINGS_DIR, os.path.basename(rec['file'])))
        
        # malorca
        sampling_rate, wav_data = wavfile.read(os.path.join(RECORDINGS_DIR, rec['file']))
        meta_data[i]['audio'] = {
            # you have to store normalized audio (-1;1) to make Audio class work
            "array" : np.array(normalize_audio(wav_data), dtype=np.float32).tolist(),
            "sampling_rate" : sampling_rate
        }

    # save the metadata with raw audio and special name
    out=json.dumps(meta_data,ensure_ascii=False)
    name = "metadata_with_raw_audio.json"
    if (os.path.exists(os.path.join(SAVE_DIR,"metadata_with_raw_audio.json"))):
        print("metadata_with_raw_audio.json already exists, creating a new one with a timestamp")
        name = f"metadata_with_raw_audio{time.time()}.json"
    
    with open(os.path.join(SAVE_DIR,name),"w") as f:
        f.write(out)