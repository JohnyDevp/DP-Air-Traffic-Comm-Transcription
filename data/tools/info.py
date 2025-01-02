import os, sys
from tqdm import tqdm
from glob import glob
import json
from pydub import AudioSegment

PREFIX="../"
# FOLDERS=["apimod"]
FOLDERS=[]
for path, folders, files in os.walk('../'):
    if path == '../':
        FOLDERS.extend(folders)
        break

# FILES=[] # not implemented

DISK_ROOT_PATH = '/run/media/johnny/31c5407a-2da6-4ef8-95ec-d294c1afec38/'
info = {}
for folder in FOLDERS:
    print(folder,file=sys.stderr)
    folder_path = os.path.join(PREFIX,folder)
    info[folder] = {}
    for file in glob(folder_path+"/*metadata*.json"):
        print(file,file=sys.stderr)
        filename = os.path.basename(file)
        data = json.load(open(file,'r'))
        
        total_files = len(data)
        total_length_seconds = 0
        for utter in tqdm(data):
            wav_path = os.path.join(DISK_ROOT_PATH,utter['audio'])
            wav_data = AudioSegment.from_wav(wav_path)
            total_length_seconds += len(wav_data) / 1000
        hours = int(total_length_seconds / 3600)
        minutes = int(total_length_seconds / 60) - hours * 60
        seconds = int(total_length_seconds ) - hours *3600 - minutes*60
        info[folder][filename] = {'total_files':total_files, 'hours': hours,
                                    'minutes':minutes, 'seconds':seconds}

for folder in info:
    print(folder)
    for meta in info[folder]:
        data = info[folder][meta]
        print('  |---',meta)
        print('       | ** total files:', data['total_files'])
        print(f"       | ** hours: {data['hours']}")
        print(f"       | ** minutes: {data['minutes']}")
        print(f"       | ** seconds: {data['seconds']}")

