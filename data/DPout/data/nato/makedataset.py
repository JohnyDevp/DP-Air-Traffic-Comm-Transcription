from datasets import load_dataset, Dataset, DatasetDict, Audio
import json

# the path to the disk with datasets of wavs and etc..
#======================================================
# CHANGE THIS TO YOUR DISK PATH
DISK_DIR="/run/media/johnny/31c5407a-2da6-4ef8-95ec-d294c1afec38/"
#======================================================

METADATA_PATH_TRAIN=["./metadata_CA_train.json","./metadata_DE_train.json","./metadata_NL_train.json","./metadata_UK_train.json"]
METADATA_PATH_TEST=["./metadata_CA_test.json","./metadata_DE_test.json","./metadata_NL_test.json","./metadata_UK_test.json"]

DATASET_SAVE_PATH_TRAIN="./nato_train_ds"
DATASET_SAVE_PATH_TEST="./nato_test_ds"

for meta,out in zip([METADATA_PATH_TRAIN, METADATA_PATH_TEST],[DATASET_SAVE_PATH_TRAIN, DATASET_SAVE_PATH_TEST]):
    dataset = load_dataset("json", data_files=meta,split="train")
    dataset = dataset.map(lambda x: {"audio": DISK_DIR + x["audio"]}, remove_columns=["audio"])
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset.save_to_disk(out)
    