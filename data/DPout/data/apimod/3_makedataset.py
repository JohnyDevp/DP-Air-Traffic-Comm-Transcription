from datasets import load_dataset, Dataset, DatasetDict, Audio
import json
import sys

# the path to the disk with datasets of wavs and etc..
#======================================================
# CHANGE THIS TO YOUR DISK PATH
if len(sys.argv) > 1:
    DISK_DIR = sys.argv[1]
else:
    DISK_DIR=""
#======================================================
METADATA_PATH="./metadata_train.json"
DATASET_SAVE_PATH="./apimod_train_ds"

dataset = load_dataset("json", data_files=METADATA_PATH,split="train")

# set properly path to the recordings according to the current disk path
dataset = dataset.map(lambda x: {"audio": DISK_DIR + x["audio"]}, remove_columns=["audio"])

# load the audio (from disk)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# save the dataset
dataset.save_to_disk(DATASET_SAVE_PATH)