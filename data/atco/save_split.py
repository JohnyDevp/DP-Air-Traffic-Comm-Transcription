from email.mime import audio
import os,json

metadata = [
    'metadata_en_ruzyne_test.json',
    'metadata_en_stefanik_test.json',
    'metadata_en_zurich_test.json',
    'metadata_en_train.json',
    'metadata_fr_test.json',
    'metadata_fr_train.json',
    'metadata_other_lang_test.json',
]
out_split = {}
for file in metadata:
    data = json.load(open(file, 'r'))
    out_split[file] = []
    for item in data:
        audio_path = item['audio']
        if ('segidx0' in audio_path):
            audio_path = audio_path.split('.wav')[0] + '.wav'
        if ('segidx' in audio_path and 'segidx0' not in audio_path):
            continue
        out_split[file].append(audio_path)

with open('split_atco.json', 'w') as f:
    json.dump(out_split, f, indent=4)