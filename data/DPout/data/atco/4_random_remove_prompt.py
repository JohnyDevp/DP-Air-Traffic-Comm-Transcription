import json
import random, sys

file= json.load(open("metadata_en_train.json"))

# remove about 5% prompts
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

with open("metadata_en_train.json", "w") as f:
    json.dump(file, f, indent=4, ensure_ascii=False)