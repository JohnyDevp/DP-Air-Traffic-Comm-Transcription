from datasets import load_dataset, Dataset, DatasetDict, Audio
import json,os,sys

# the path to the disk with datasets of wavs and etc..
#======================================================
# CHANGE THIS TO YOUR DISK PATH
if len(sys.argv) > 1:
    DISK_ROOT = sys.argv[1]
else:
    DISK_ROOT=""
#======================================================

METADATA_PATH       =   ["./metadata_hwir_fr_test.json","./metadata_hwir_gr_test.json",'./metadata_hwir_sp_test.json','./metadata_hwir_train.json']
DATASET_SAVE_PATH   =   ["./hwir_fr_test_ds","./hwir_gr_test_ds","./hwir_sp_test_ds","./hwir_train_ds"]

for i in range(len(METADATA_PATH)):
    dataset = load_dataset("json", data_files=METADATA_PATH[i], split="train")

    # set properly path to the recordings according to the current disk path
    dataset = dataset.map(lambda x: {"audio": os.path.join(DISK_ROOT, x["audio"])}, remove_columns=["audio"])

    # load the audio (from disk)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    # save the dataset
    dataset.save_to_disk(DATASET_SAVE_PATH[i])