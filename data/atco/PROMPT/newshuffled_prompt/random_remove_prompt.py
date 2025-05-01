import json
import random

file= json.load(open("metadata_en_train.json"))
for _ in range(75):
    while True:
        idx =random.randint(0,len(file)-1)
        item = file[idx]
        if item['prompt_fullts_AG'].strip() == "": continue
        else:
            item['prompt_fullts_AG'] = ""
            item['prompt_fullts_AG_4B'] = ""
            item['prompt_fullts_AG_40B'] = ""
            item['prompt_fullts_AG_50CZB'] = ""
            break