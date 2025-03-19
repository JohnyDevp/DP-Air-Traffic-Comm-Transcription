import json,os
from bs4 import BeautifulSoup
import re
from unicodedata import normalize
import random
from tqdm import tqdm

def get_tagged_parts(xml_data, tags_to_extract):
    soup = BeautifulSoup(xml_data, "xml")
    out = {}
    for segment in soup.find_all("segment"):
        found_tags = get_tags_content_for_one_segment(segment,tags_to_extract)
        # merge the dictionaries
        for tag, content in found_tags.items():
            if tag in out:
                out[tag].extend(content)
            else:
                out[tag] = content
    return out

def get_tags_content_for_one_segment(segment, tags_to_extract) -> str:
    # first obtain tags from the text of the segment
    text = segment.find("text").text
    out = {}
    for tag in tags_to_extract:
        pattern = r'\['+tag+r'\](.*?)\[/'+tag+r'\]'
        out[tag] = []
        for it in re.findall(pattern, text):
            out[tag].append(re.sub(r'\s+', ' ', normalize('NFC',it)))
    
    # obtain speaker labels
    out['speaker_label'] = [segment.find("speaker_label").text]
    return out


if __name__ == '__main__':
    METADATA_PATHS = [
        'metadata_en_ruzyne_test.json', 'metadata_en_stefanik_test.json','metadata_en_zurich_test.json',
        'metadata_en_train.json',
        'metadata_fr_test.json', 'metadata_fr_train.json',
        'metadata_other_lang_test.json'
    ]
    DISK_PATH = '/run/media/johnny/31c5407a-2da6-4ef8-95ec-d294c1afec38/'
    # where to look for files that belongs to the one recording (like .seg, .info,...)
    FOLDERS_TO_SEARCH = ['ATCO2-ASRdataset-v1_final/DATA_nonEN-original','ATCO2-ASRdataset-v1_final/DATA-original']
    SAVE_FOLDER = './PROMPT/'
 
    for file in METADATA_PATHS:
        with open(file,'r') as f:
            js_file = json.load(f)
            for meta in tqdm(js_file):
                # extract the recording file name without extension
                rec_file = os.path.basename(meta['audio']).split('.wav')[0]
                for fl in FOLDERS_TO_SEARCH:
                    path = os.path.join(DISK_PATH,fl,rec_file+'.wav')
                    if os.path.exists(path):
                        break
                
                # open xml and info files
                xml_file = path.replace('.wav','.xml')
                # info_file = path.replace('.wav','.info')
                
                # from xml file obtain all callsigns
                with open(xml_file,'r') as f:
                    xml_data = f.read()
                    out = get_tagged_parts(xml_data, ['#callsign'])
                
                meta['prompt-data'] = {
                    'callsigns': list(set(out['#callsign'])),
                    'speaker_labels': list(set(out['speaker_label'])),
                    'waypoints': meta['prompt']['waypoints'],
                    'short_callsigns': meta['prompt']['short_callsigns'],
                    'long_callsigns': meta['prompt']['long_callsigns']
                }
                len_waypoints = len(meta['prompt-data']['waypoints'])
                len_long_callsign = len(meta['prompt-data']['long_callsigns'])
                meta['prompt_fullts'] = ', '.join(meta['prompt-data']['callsigns']) + ', ' + \
                    ', '.join(random.sample(meta['prompt-data']['waypoints'],min(3,len_waypoints))) + ', ' + \
                    ', '.join(random.sample(meta['prompt-data']['long_callsigns'],min(3,len_long_callsign)))
                meta.pop('prompt')
            
            json.dump(js_file, open(SAVE_FOLDER+file,'w'),indent=4)
            f.close()