# {
#   "filepath":str,
#   "full_ts":str,
#   "short_ts":str,
#   "language":str     
# }

import sys
import glob as glob
import os
from unittest import result
from bs4 import BeautifulSoup
from rapidfuzz import process
import re
from tqdm import tqdm
import json
import time

number_map = {
    # English
    "zero": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7",
    "eight": "8", "nine": "9",
    
    # Teens and special numbers
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16",
    "seventeen": "17", "eighteen": "18", "nineteen": "19",

    # Tens
    "twenty": "20", "thirty": "30", "forty": "40", "fourty": "40",
    "fifty": "50", "sixty": "60", "seventy": "70", 
    "eighty": "80", "ninety": "90",
    
    
    # Czech
    "nula": "0", "jedna": "1", "dva": "2", "tři": "3",
    "čtyři": "4", "pět": "5", "šest": "6", "sedm": "7",
    "osm": "8", "devět": "9",

    "deset": "10", "jedenáct": "11", "dvanáct": "12", "třináct": "13",
    "čtrnáct": "14", "patnáct": "15", "šestnáct": "16",
    "sedmnáct": "17", "osmnáct": "18", "devatenáct": "19",
    
    "dvacet": "20", "třicet": "30", "čtyřicet": "40",
    "padesát": "50", "šedesát": "60", "sedmdesát": "70",
    "osmdesát": "80", "devadesát": "90",
    
    # German
    "null": "0", "eins": "1", "zwei": "2", "drei": "3",
    "vier": "4", "fünf": "5", "sechs": "6", "sieben": "7",
    "acht": "8", "neun": "9",
    
    "zehn": "10", "elf": "11", "zwölf": "12", "dreizehn": "13",
    "vierzehn": "14", "fünfzehn": "15", "sechzehn": "16",
    "siebzehn": "17", "achtzehn": "18", "neunzehn": "19",
    
    "zwanzig": "20", "dreisig": "30", "vierzig": "40",
    "fünfzig": "50", "sechzig": "60", "siebzig": "70",
    "achtzig": "80", "neunzig": "90",
    
    # the rest....
    "hundred": "00", "thousand": "000",
    "hundert": "00", "tausend": "000",
    "sto": "00", "tisíc": "000",
    
    "decimal": ".", "point": "."
}

aviation_map = {
    "alpha": "A", "bravo": "B", "charlie": "C", "charly":"C", "delta": "D",
    "echo": "E", "foxtrot": "F", "fox":"F", "golf": "G", "hotel": "H",
    "india": "I", "juliet": "J", "juliett":"J", "kilo": "K", "lima": "L",
    "mike": "M", "november": "N", "oscar": "O", "papa": "P",
    "quebec": "Q", "romeo": "R", "sierra": "S", "tango": "T",
    "uniform": "U", "victor": "V", "whiskey": "W", "whisky": "W",
    "x-ray": "X", "xray":"X", "yankee": "Y", "zulu": "Z", "zoulou": "Z"
}
leave_untouch_words = ["and", "on", "or"]

def process_tag_content(text, what : str ="alphanum",cutoff=None):
    # what : str = "alphanum" | "num" | "alpha"
    match what:
        case "alphanum":
            combined_map = {**number_map, **aviation_map}
            score_cutoff = 93
        case "num":
            combined_map = number_map
            score_cutoff = 70
        case "alpha":
            combined_map = aviation_map
            score_cutoff = 70
    
    if (cutoff):
        score_cutoff = cutoff
        
    # Process the words
    result = []
    current_transcript = ""
    
    tagwords = text.strip().split(' ')
    for word in tagwords:
        # Find the closest match from the combined_map keys
            bestmatch = process.extractOne(word.lower(), combined_map.keys(), score_cutoff=score_cutoff)
            if bestmatch:
                current_transcript += combined_map[bestmatch[0]]  # Add the matched transcription
            else:
                if current_transcript:  # If there is an ongoing transcription chunk
                    result.append(current_transcript)
                    current_transcript = ""
                result.append(word)  # Append the normal word as-is
    
    # append the last chunk, if exists
    if current_transcript:
        result.append(current_transcript)
        
    return ' '.join(result)

def make_vocab(info_path):
    vocab = {}
    with open(info_path, "r") as f:
        lines = f.readlines()[8:] # first 8 lines are not needed
        for line in lines:
            line_split = line.split(':')
            if (len(line_split) != 2 or line_split[0].strip().__len__() < 4 or line_split[1].split(' ').__len__() > 10):
                continue
            code = line.split(":")[0].lower().strip()
            full = line.split(":")[1].lower().strip()
            words = full.split(' ')
            
            if (words[0] in aviation_map.keys() or words[0] in number_map.keys()):
                # it means the code consists just of normal airtraffic alphabet
                continue
            words_of_airport_icao = ' '.join(words[0:int(len(words) - (len(code)-3))]).strip()
            vocab[words_of_airport_icao] = code[0:3] 
    return vocab

def process_callsign_from_info(callsign_plain : str, airport_vocab : dict):
    partly_processed_callsign = process_tag_content(callsign_plain, "alphanum")
    # we assume that the last word is now built from numbers and letters recognized 
    # from air traffic alphabet and numbers
    
    # we will try to find the airport code in the info file
    airport_name = ' '.join(partly_processed_callsign.strip().lower().split(' ')[0:-1]).strip()
    if (airport_name in airport_vocab and airport_name.strip() != ""):
        airport_code = airport_vocab[airport_name]
        return (airport_code + partly_processed_callsign.split(' ')[-1]).upper()
    else:
        return partly_processed_callsign.strip()

vocab_global = {}
def get_shortts(wav_full_path_current_disk, xml_data) -> str:
    # get short ts
    soup = BeautifulSoup(xml_data, "xml")
    shortts = ""
    for segment in soup.find_all("segment"):
        # obtain the text tag amd remove the <text> tag from it
        text_w_tag = segment.find("text")
        seg_text = text_w_tag.get_text(separator=" ", strip=True)
        
        # ensure tags contains everything they can (meaning, multiple same tags going one after another are merged)
        pattern = r'\[\/#(\w+)\]\s*\[#\1\]'
        seg_text = re.sub(pattern, ' ', seg_text)
        
        # go through tags callsign and value and shorten them
        searched_tags = ['callsign', 'value']
        for tag in searched_tags:
            pattern = rf"\[#({tag})\](.*?)\[/#{tag}\]"
            if tag == "callsign":
                info_path = wav_full_path_current_disk.replace(".wav", ".info")
                vocab = {}
                if (os.path.exists(info_path)):
                    vocab = make_vocab(info_path)
                    vocab_global = vocab # just to give vocab for the wavs that do not have info file
                else:
                    vocab = vocab_global 
                seg_text=re.sub(pattern, lambda x: process_callsign_from_info(x.group(2), vocab),seg_text)
                
            elif tag == "value":
                seg_text=re.sub(pattern, lambda x: process_tag_content(x.group(2), "num",cutoff=80),seg_text)
        
        # now replace remove all other still existing tags
        # and multiple spaces
        pattern = r'\[[^\]]*\]'
        seg_text = re.sub(pattern,"",seg_text)
        pattern = r'\s+'
        seg_text = re.sub(pattern," ",seg_text)
        
        # finnaly go again with alphanum processing
        seg_text = process_tag_content(seg_text, "alphanum")
        
        # append
        shortts += seg_text + '\n'

    return shortts

def get_plain_text_from_segment(tag_text):
    pattern = r'\[[^\]]*\]'
    return re.sub(pattern,"",tag_text)

def get_fullts(xml_data):
    soup = BeautifulSoup(xml_data, "xml")
    fullts = ""
    for segment in soup.find_all("segment"):
        text_tag = segment.find("text")
        fullts += get_plain_text_from_segment(text_tag.get_text(separator=" ", strip=True))
        fullts += "\n"
    return fullts

def get_prompt(xml_file):
    return None

def makemetadata(wav_listings_path : str, disk_path_tb_excluded : str, out_file_name : str):
    out_data = []
    
    for wav_file_path in tqdm(wav_listings_path):
        wav_full_path_current_disk = os.path.join(disk_path_tb_excluded, wav_file_path)               
        
        # obtain the xml file (check if it exists) with the transcription
        xml_file = wav_full_path_current_disk.replace(".wav", ".xml")
        
        if (not os.path.exists(xml_file)):
            print(f"File {xml_file} does not exist")
            continue
        
        with open(xml_file, "r") as f:  
            xml_data = f.read()
            # extract the full ts            
            full_ts = get_fullts(xml_data)
            if (full_ts.strip() == ""):
                print(f"File {xml_file} has empty transcription, skipping", file=sys.stderr)
                continue
            
            short_ts = get_shortts(wav_full_path_current_disk, xml_data)

            prompt = get_prompt(xml_file)
            
            out_data.append({
                "audio": wav_file_path, # just want the path on the disk, not whole on the pc
                "full_ts": full_ts,
                "short_ts": short_ts,
                "prompt": prompt,
            })
    
    # save the metadata, ensure no overwriting of previously created metadata
    out=json.dumps(out_data,indent=4,ensure_ascii=False)
    name = out_file_name
    if (os.path.exists(name)):
        print(f"{name} already exists, creating a new one with a timestamp")
        name = f"{name}{time.time()}.json"
    with open(name,"w") as f:
        f.write(out) 
        f.close()

def split_wav_files_en(root : str) -> list:
    # counts for datasets ruzyne and stefanik - from each is used for test 350 files
    # and whole LZSH_Zurich is used for test
    
    count_test_ruzyne = 0
    count_test_stefanik = 0
    
    result_test_stefanik = []
    result_test_ruzyne = []
    result_test_zurich = []
    result_train = []
    for file in glob.glob(root +"/*.wav"):
        if (count_test_ruzyne < 350 and "LKPR_RUZYNE" in file):
            result_test_ruzyne.append(file)
            count_test_ruzyne += 1
        elif (count_test_stefanik < 350 and "LZIB_STEFANIK" in file):
            result_test_stefanik.append(file)
            count_test_stefanik += 1
        elif ("LSZH_ZURICH" in file):
            result_test_zurich.append(file)
        else:
            result_train.append(file)
    return [result_train, result_test_ruzyne,result_test_stefanik,result_test_zurich]

if __name__ == "__main__":
    ROOT_DIR_EN="/run/media/johnny/31c5407a-2da6-4ef8-95ec-d294c1afec38/ATCO2-ASRdataset-v1_final/DATA"
    files_train, files_test_ruzyne,files_test_stefanik,files_test_zurich = split_wav_files_en(ROOT_DIR_EN)
    makemetadata(files_train, disk_path_tb_excluded="/run/media/johnny/31c5407a-2da6-4ef8-95ec-d294c1afec38", out_file_name="metadata_en_train.json")
    makemetadata(files_test_ruzyne, disk_path_tb_excluded="/run/media/johnny/31c5407a-2da6-4ef8-95ec-d294c1afec38",out_file_name="metadata_en_ruzyne_test.json")
    makemetadata(files_test_stefanik, disk_path_tb_excluded="/run/media/johnny/31c5407a-2da6-4ef8-95ec-d294c1afec38",out_file_name="metadata_en_stefanik_test.json")
    makemetadata(files_test_zurich, disk_path_tb_excluded="/run/media/johnny/31c5407a-2da6-4ef8-95ec-d294c1afec38",out_file_name="metadata_en_zurich_test.json")
    # print(len(files_train), len(files_test_ruzyne), len(files_test_stefanik), len(files_test_zurich))
    # ROOT_DIR_NONEN="/run/media/johnny/31c5407a-2da6-4ef8-95ec-d294c1afec38/ATCO2-ASRdataset-v1_final/DATA_nonEN"
    # wav_files_nonen = get_wav_file_names(ROOT_DIR_NONEN)
   
    