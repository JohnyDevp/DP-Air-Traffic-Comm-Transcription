from operator import contains
import os, json, re
from tqdm import tqdm
from bs4 import BeautifulSoup

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
    
    LOOKS_FOR = ['runway','waypoint']
    
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
                xml_file_content = open(xml_file,'r').read()
                content = BeautifulSoup(xml_file_content, 'xml')
                for word in LOOKS_FOR:
                    for segment in content.find_all('segment'):
                        segment_text = segment.find('text').text
                        if (re.search(word, segment_text) is not None):
                            print(f'{xml_file}')
                        # speaker_label : str = segment.find('speaker_label').text
                        # if (re.search('callsign', segment_text) is not None and 'UNK'.lower() in speaker_label.lower()):
                        #     print(f'{xml_file}')