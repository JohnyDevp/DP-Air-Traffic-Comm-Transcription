import os, glob, json
from scipy.io import wavfile
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict, Audio

def normalize_audio(audio_array): # necessary to normalize the audio (datasets expect values between -1 and 1)
    return audio_array / np.max(np.abs(audio_array))

# function overtaken from whisper finetuning script
def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids **** CHANGED FROM **sentence** TO **transcription**
    batch["labels"] = tokenizer(batch["transcription"]).input_ids
    return batch


if __name__ == "__main__":

    RECORDINGS_DIR="apimod/01_01_EL_LN_UJ_VV_YADA"
    META_FILE="apimod/01_01_EL_LN_UJ_VV_YADA/metadata.json"


    # load the metadata
    meta_data = json.load(open(META_FILE, mode='r'))

    # load the wav files to the metadata
    for i, rec in enumerate(meta_data):
        sampling_rate, wav_data = wavfile.read(os.path.join(RECORDINGS_DIR, os.path.basename(rec['file'])))
        meta_data[i]['audio'] = {
            # you have to store normalized audio (-1;1) to make Audio class work
            "array" : np.array(normalize_audio(wav_data), dtype=np.float32).tolist(),
            "sampling_rate" : sampling_rate
        }

    # load it to the dataset and split to test and train
    dd = Dataset.from_list(meta_data)
    dd_dsdict = DatasetDict({"train": dd})
    # dd_dsdict['train'].train_test_split(test_size=0.2,seed=42)
    
    # ensure 16000 sampling rate
    dd_dsdict = dd_dsdict.cast_column("audio",Audio(sampling_rate=16000))
    
    # load tokenizer and feature extractor
    from transformers import WhisperFeatureExtractor, WhisperTokenizer
    feature_extractor = WhisperFeatureExtractor.from_pretrained("opeanai/whisper-medium")
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-medium", task="transcribe")
    
    # do resampling by reloading of the data, and change the content of the dataset to obtained FEATURES and LABELS
    dd_dsdict = dd_dsdict.map(prepare_dataset, remove_columns=dd_dsdict.column_names["train"], num_proc=4)
    
    # save the dataset
    dd_dsdict.save_to_disk(os.path.join(RECORDINGS_DIR, "apimod.dataset.training"))