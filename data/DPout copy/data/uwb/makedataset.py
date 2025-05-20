from datasets import load_dataset, Dataset, DatasetDict, Audio
import json

# the path to the disk with datasets of wavs and etc..
#======================================================
# CHANGE THIS TO YOUR DISK PATH
DISK_ROOT=""
#======================================================

METADATA_PATH=["./metadata_train.json", "./metadata_test.json"]
DATASET_SAVE_PATH=["./uwb_train_ds","./uwb_test_ds"]

for meta,out in zip(METADATA_PATH,DATASET_SAVE_PATH):
    dataset = load_dataset("json", data_files=meta,split="train")
    dataset = dataset.map(lambda x: {"audio": DISK_ROOT + x["audio"]}, remove_columns=["audio"])
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset.save_to_disk(out)