import json, os, sys, random, re

FOLDERS_TO_SEARCH = ['ATCO2-ASRdataset-v1_final/DATA_nonEN-original','ATCO2-ASRdataset-v1_final/DATA-original']
DISK_PATH='/run/media/johnny/31c5407a-2da6-4ef8-95ec-d294c1afec38/'

file = json.load(open("metadata_en_zurich_test.json"))
for item in file:
    # extract the recording file name without extension
    rec_file = os.path.basename(item['audio']).split('.wav')[0]
    # build up the correct path to the current disk, where the file should be located
    path = os.path.join(DISK_PATH,item['audio'])
    for fl in FOLDERS_TO_SEARCH:
        p = os.path.join(DISK_PATH,fl,rec_file+'.wav')
        if os.path.exists(p):
            path = p
            break
    if item['prompt_fullts_AG'].strip() == "":
        print(path.replace('wav','xml'))