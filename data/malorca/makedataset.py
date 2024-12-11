from datasets import load_dataset, Dataset, DatasetDict, Audio
import json

# the path to the disk with datasets of wavs and etc..
DISK_DIR="/run/media/johnny/31c5407a-2da6-4ef8-95ec-d294c1afec38/"
METADATA_PATH="./metadata_test.json"
DATASET_SAVE_PATH="./malorca_test_ds"

dataset = load_dataset("json", data_files=METADATA_PATH)

# set properly path to the recordings according to the current disk path
dataset = dataset.map(lambda x: {"audio": DISK_DIR + x["audio"]}, remove_columns=["audio"])

# load the audio (from disk)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# save the dataset
dataset.save_to_disk(DATASET_SAVE_PATH)