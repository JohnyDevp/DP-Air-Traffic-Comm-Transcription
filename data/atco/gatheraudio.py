from decimal import ROUND_UP
import time
from scipy.io import wavfile
from glob import glob
import os
import json
import numpy as np
# this is to be setup
DISK_PATH = '/run/media/johnny/31c5407a-2da6-4ef8-95ec-d294c1afec38/'
METADATA_FILE = 'metadata_en_ruzyne_test.json'

time_idx_dict = {}
if __name__ == '__main__':
    # load the meta file 
    meta_json = json.load(open(METADATA_FILE))
    
    # loop through and gather the audios length, saving it to the dict
    for item in meta_json:
        rate,data = wavfile.read(os.path.join(DISK_PATH, item['audio']))
        duration = int(np.ceil(len(data)/rate))
        if duration in time_idx_dict: time_idx_dict[duration] += [item]
        else: time_idx_dict[duration] = [item]
    
    # sort the dict
    time_idx_dict = dict(sorted(time_idx_dict.items(),reverse=True))
    
    result_dict= []
    current_time = 0
    used_items = 0
    
    # loop through the dict and gather the audios into subgroups under 30 seconds
    while used_items < len(meta_json):
        current_gathered_items = {
            'audio':[],
            'full_ts':"",
            'short_ts':"",
            'prompt':[]
        }
        for key in time_idx_dict:
            if int(key) + current_time > 30 or time_idx_dict.get(key) == []: continue
            item = time_idx_dict[key].pop(0)
            current_gathered_items['audio'] += [item['audio']]
            current_gathered_items['full_ts'] += [item['full_ts']]
            current_gathered_items['short_ts'] += [item['short_ts']]
            current_gathered_items['prompt'] += [item['prompt']]
            current_time += int(key)
            used_items += 1
            
        result_dict.append(current_gathered_items)
        current_time = 0

    json.dump(result_dict, open('gathered.json','w'))
        
        

    
