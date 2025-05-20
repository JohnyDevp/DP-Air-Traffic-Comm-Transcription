from datasets import load_dataset, Dataset, DatasetDict, Audio
import json, sys

# the path to the disk with datasets of wavs and etc..
#======================================================
# CHANGE THIS TO YOUR DISK PATH
if len(sys.argv) > 1:
    DISK_ROOT = sys.argv[1]
else:
    DISK_ROOT=""
#======================================================

METADATA_PATH_TRAIN=["./metadata_CA_train.json","./metadata_DE_train.json","./metadata_NL_train.json","./metadata_UK_train.json"]
METADATA_PATH_TEST=["./metadata_CA_test.json","./metadata_DE_test.json","./metadata_NL_test.json","./metadata_UK_test.json"]

DATASET_SAVE_PATH_TRAIN="./nato_train_ds"
DATASET_SAVE_PATH_TEST="./nato_test_ds"

for meta,out in zip([METADATA_PATH_TRAIN, METADATA_PATH_TEST],[DATASET_SAVE_PATH_TRAIN, DATASET_SAVE_PATH_TEST]):
    dataset = load_dataset("json", data_files=meta,split="train")
    dataset = dataset.map(lambda x: {"audio": DISK_ROOT + x["audio"]}, remove_columns=["audio"])
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset.save_to_disk(out)
    