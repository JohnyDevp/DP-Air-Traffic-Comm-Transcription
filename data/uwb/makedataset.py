from datasets import load_dataset, Dataset, DatasetDict, Audio
import json

# the path to the disk with datasets of wavs and etc..
DISK_DIR="/run/media/johnny/31c5407a-2da6-4ef8-95ec-d294c1afec38/"
METADATA_PATH=["./metadata_train.json", "./metadata_test.json"]
DATASET_SAVE_PATH=["./uwb_train_ds","./uwb_test_ds"]

for meta,out in zip(METADATA_PATH,DATASET_SAVE_PATH):
    dataset = load_dataset("json", data_files=meta,split="train")
    dataset = dataset.map(lambda x: {"audio": DISK_DIR + x["audio"]}, remove_columns=["audio"])
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset.save_to_disk(out)