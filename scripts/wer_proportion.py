import os
import re, json
import sys
from bs4 import BeautifulSoup, ResultSet


meta_files=['metadata_en_ruzyne_test.json','metadata_en_stefanik_test.json','metadata_en_zurich_test.json','metadata_en_train.json']
ROOT = '../data/atco/PROMPT/'
ROOT_DISK ='/run/media/johnny/31c5407a-2da6-4ef8-95ec-d294c1afec38/'
FOLDERS_TO_SEARCH = ['ATCO2-ASRdataset-v1_final/DATA_nonEN-original','ATCO2-ASRdataset-v1_final/DATA-original']

portioned = []
everything_chars_length = 0
everything_word_length = 0
for file in meta_files:
    file_path = os.path.join(ROOT, file)
    js = json.load(open(file_path))
    xml_files = [item['audio'].replace('wav','xml') for item in js ]

    fullts_total_chars_length = 0
    fullts_total_word_length = 0
    callsigns_chars_length = 0
    callsigns_word_length = 0
    percentage_total = 0.0
    weighted_percentage_total = 0.0
    weights_total = 0
    total_records =0
    total_callsigns = 0
    cal_write = open(file,'w')
    
    for xml_file in xml_files:
        if ('segidx' in xml_file and 'segidx0' not in xml_file): 
            print(f'File {xml_file} already processed',file=sys.stderr)
            continue
        elif ('segidx0' in xml_file):
            xml_file_basename = os.path.basename(xml_file).split('.xml')[0]
            # build up the correct path to the current disk, where the file should be located
            # portioned.append(xml_file_basename)
            for fl in FOLDERS_TO_SEARCH:
                p = os.path.join(ROOT_DISK,fl,xml_file_basename+'.xml')
                if os.path.exists(p):
                    xml_file = p
                    break
        
        total_records += 1
        
        xml_content = open(os.path.join(ROOT_DISK,xml_file),'r').read()
        bs=BeautifulSoup(xml_content,features='xml')
        full_ts_with_tags = ''    
        for segment in bs.find_all('segment'):
            full_ts_with_tags +=re.sub(r'\[\/#(\w+)\]\s*\[#\1\]',' ',segment.find('text').text)
        
        # now obtain all words in callsigns
        pattern= r'\[#callsign\](.*?)\[/#callsign\]'
        all_callsigns = re.findall(pattern, full_ts_with_tags)
        total_callsigns += len(all_callsigns) 
        all_callsigns_chars_length = 0
        all_callsigns_word_length = 0
        cal_write.write(f'{xml_file}\t{len(all_callsigns)}\n')
        for callsign in all_callsigns:
            cal_write.write(f'\t{callsign}\n')
            # normalize callsign
            callsign = re.sub(r'\s+', ' ', callsign).strip()
            all_callsigns_chars_length += len(callsign)
            all_callsigns_word_length += len(callsign.split())
        cal_write.write('\n')
        # remove all callsigns tags with its content
        # full_ts_without_callsigns = re.sub(pattern, ' ', full_ts_with_tags)
        # now remove all other tags
        full_ts_clean = re.sub(r'\[[^\]]*\]',' ',full_ts_with_tags)
        # clean spaces
        full_ts_clean = re.sub(r'\s+',' ',full_ts_clean).strip()
        
        # now we can count the number of words
        full_ts_clean_word_length = len(full_ts_clean.split())
        # now we can count the number of characters
        full_ts_clean_chars_length = len(full_ts_clean)
        
        # SUM EVERYTHING SO FAR
        percentage_total += all_callsigns_chars_length /float(full_ts_clean_chars_length) * 100.0
        weighted_percentage_total += (all_callsigns_chars_length /float(full_ts_clean_chars_length)) * full_ts_clean_chars_length
        weights_total += full_ts_clean_chars_length
        
        fullts_total_chars_length += full_ts_clean_chars_length
        fullts_total_word_length += full_ts_clean_word_length
        callsigns_chars_length += all_callsigns_chars_length
        callsigns_word_length += all_callsigns_word_length
        
    # print the results
    print(f'File: {file}')
    print(f'Total records: {total_records}')
    print(f'Full TS without callsigns: {fullts_total_chars_length} chars, {fullts_total_word_length} words')
    print(f'Total callsigns: {total_callsigns}')
    print(f'Calls signs: {callsigns_chars_length} chars, {callsigns_word_length} words')
    print(f'AVG Percentage of callsigns: {(percentage_total/len(xml_files)):.2f}%; Weighted: {(weighted_percentage_total/weights_total * 100):.2f}%')     
    print(f'Total percentage in chars: {callsigns_chars_length / float(fullts_total_chars_length) * 100.0:.2f}%')
    print(f'Total percentage in words: {callsigns_word_length / float(fullts_total_word_length) * 100.0:.2f}%')
    print()
            
