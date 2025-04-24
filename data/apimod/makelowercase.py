import json 
import re

file='../nato/metadata_CA_train.json'
with open(file, 'r', encoding='utf-8') as f:
    data = json.load(f)
    for item in data:
        for key in item:
            if key != 'audio' and isinstance(item[key], str):
                item[key] = re.sub(r'\s+',' ',item[key].lower()).strip()
                
    with open(file, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)