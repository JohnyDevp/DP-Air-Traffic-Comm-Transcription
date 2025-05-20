import sys
import glob as glob
import os
from bs4 import BeautifulSoup
from rapidfuzz import process
import re
from tqdm import tqdm
import json
import time
from pydub import AudioSegment


units = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
tens = ['20', '30', '40', '50', '60', '70', '80', '90'] 
teens = ['11', '12', '13', '14', '15', '16', '17', '18', '19']

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
    "sto": "00", "set":"00", "tisíc": "000",
    
    "decimal": ".", "point": "."
}

aviation_map = {
    "alpha": "A", "alfa": "A", "bravo": "B", "charlie": "C", "charly":"C", "delta": "D",
    "echo": "E", "foxtrot": "F", "fox":"F", "golf": "G", "hotel": "H",
    "india": "I", "juliet": "J", "juliett":"J", "kilo": "K", "lima": "L",
    "mike": "M", "november": "N", "oscar": "O", "papa": "P",
    "quebec": "Q", "romeo": "R", "sierra": "S", "siera": "S", "tango": "T",
    "uniform": "U", "victor": "V", "viktor": "V", "whiskey": "W", "whisky": "W",
    "x-ray": "X", "xray":"X", "yankee": "Y", "yanke": "Y", "zulu": "Z", "zoulou": "Z"
}
leave_untouch_words = ["and", "on", "or"]

def parse_forward(words,idx):
    wtbp = []
    myidx = 0
    while idx + myidx < len(words):
        word = process.extractOne(words[idx + myidx].lower(), number_map.keys(), score_cutoff=80)
        if words[idx + myidx] == 'and': #skip
            myidx += 1
        elif word and number_map[word[0]] != '.':
            wtbp.append(number_map[word[0]])
            myidx += 1
        else:
            break
    
    # check for little chance, that we have only "hundred" instead of "one hundred"
    if len(wtbp) >= 1 and wtbp[0] in ['00','000']:
        wtbp = ['1'] + wtbp
    
    l = len(wtbp)
    if l == 0:
        return False, "", idx
    elif l == 2 and wtbp[0] in tens and wtbp[1] in units: # 20 1 -> 21
        return True, str(int(wtbp[0]) + int(wtbp[1])), idx + myidx
    elif l == 2 and wtbp[0] in [*units,*tens,*teens,'10'] and wtbp[1] in ['00','000']: # 2000
        return True, str(wtbp[0] + wtbp[1]), idx + myidx
    elif l == 3 and wtbp[0] in units and wtbp[1] in ['00','000'] and (wtbp[2] in [*units,*tens,*teens,'10']): # one [thousand, hundred] one => 1001 (101)
        return True, str(int(wtbp[0]+wtbp[1]) + int(wtbp[2])), idx + myidx
    elif l == 4 and wtbp[0] in units and wtbp[1] in ['00','000'] and wtbp[2] in tens and wtbp[3] in units: # one [thousand, hundred] twenty one => 1021
        return True, str(int(wtbp[0]+wtbp[1]) + int(wtbp[2]) + int(wtbp[3])), idx + myidx
    elif l == 4 and wtbp[0] in [*units,'10'] and wtbp[1] in ['000'] and wtbp[2] in units and wtbp[3] in ['00']:
        return True, str(int(wtbp[0]+wtbp[1]) + int(wtbp[2]+wtbp[3])), idx + myidx
    elif l == 5 and wtbp[0] in [*units,'10'] and wtbp[1] in ['000'] and wtbp[2] in units and wtbp[3] in ['00'] and wtbp[4] in [*units,*tens,*teens,'10']:
        return True, str(int(wtbp[0]+wtbp[1]) + int(wtbp[2]+wtbp[3]) + int(wtbp[4])), idx + myidx
    elif l == 6 and wtbp[0] in [*units,'10',*teens,*tens] and wtbp[1] in ['000'] and wtbp[2] in units and wtbp[3] in ['00'] and wtbp[4] in tens and wtbp[5] in units:
        return True, str(int(wtbp[0]+wtbp[1]) + int(wtbp[2]+wtbp[3]) + int(wtbp[4]) + int(wtbp[5])), idx + myidx
    else:
        return False, "", idx
   
def process_tag_content(full_ts, what : str ="alphanum",cutoff=None):
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
    
    # split the words and also punctuation separately
    reg = re.compile(r"\w+|[.,!?]")
    words : list[str] = reg.findall(full_ts)
    
    idx = 0
    while idx < len(words):
        word = words[idx]
        # Find the closest match from the combined_map keys
        bestmatch = process.extractOne(word.lower(), combined_map.keys(), score_cutoff=score_cutoff)
    
        if idx + 1 < len(words) and bestmatch and what in ["alphanum", "num"]:
            is_parsed, to_add, next_idx = parse_forward(words, idx)
            if is_parsed:
                if current_transcript:  # If there is an ongoing transcription chunk
                    result.append(current_transcript)
                    current_transcript = ""
                result.append(to_add)  # append the parsed number
                idx = next_idx
                continue          
        
        if bestmatch:
            current_transcript += combined_map[bestmatch[0]]  # Add the matched transcription
            idx += 1
            continue
            
        if current_transcript:  # If there is an ongoing transcription chunk
            result.append(current_transcript)
            current_transcript = ""
        result.append(word)  # Append the normal word as-is
        idx += 1
    
    # append the last chunk, if exists
    if current_transcript:
        result.append(current_transcript)

    result = ' '.join(result)
    # remove potential spaces between decimal points
    result = re.sub(r'(?<=\d)\s+\.\s+(?=\d)', '.', result) # 1 . 2 -> 1.2
    
    result = re.sub(r'\s+([.,!?])', r'\1', result) # remove spaces before punctuation
    return result

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
    # this function works for callsigns like Air France 1234, where the last word of digits
    # we obtain by calling process_tag_content, and Air France we translate to shortcut by 
    # looking into the info file
    
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
        seg_text = get_shortts_for_one_segment(segment,wav_full_path_current_disk)
        shortts += seg_text.strip() + '\n'

    return shortts.strip()

def get_shortts_for_one_segment(segment, wav_full_path_current_disk) -> str:
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
        
        return seg_text
    
def get_plain_text_from_segment(tag_text):
    pattern = r'\[[^\]]*\]'
    return re.sub(pattern,"",tag_text)

def get_fullts(xml_data):
    soup = BeautifulSoup(xml_data, "xml")
    fullts = ""
    for segment in soup.find_all("segment"):
        seg_text = get_fullts_for_one_segment(segment)
        fullts += seg_text.strip() + "\n"
    return fullts.strip()

def get_fullts_for_one_segment(segment) -> str:
    text_tag = segment.find("text")
    fullts = get_plain_text_from_segment(text_tag.get_text(separator=" ", strip=True))
    return fullts

def get_prompt(xml_file):
    info_file = xml_file.replace(".xml", ".info")
    if (not os.path.exists(info_file)):
        return None
    
    with open(info_file, "r") as f:
        data = f.read()
        # Extract waypoints
        waypoints_match = re.search(r"waypoints nearby: (.+)", data)
        waypoints_array = waypoints_match.group(1).split() if waypoints_match else []

        # Extract callsign shorts and long forms
        callsigns_match = re.findall(r"(\S+)\s+:\s+(.*)", data)
        call_sign_shorts = [match[0] for match in callsigns_match]
        call_sign_longs = [match[1] for match in callsigns_match]
        
    return waypoints_array, call_sign_shorts, call_sign_longs

def is_english_lang(xml_data):
    # returns True if more than 50% of the segments are in English, otherwise False
    soup = BeautifulSoup(xml_data, "xml")
    english_segments = 0
    num_of_segments = len(soup.find_all("segment"))
    for segment in soup.find_all("segment"):
        tags = segment.find('tags')
        if (tags):
            tag_en = tags.find("non_english")
            if (tag_en): # the tag is <non_english></non_english>, meaning 0 is english
                english_segments += 1 - int(tag_en.get_text())
                    
    return english_segments / num_of_segments > 0.5

def makemetadata(wav_listings_path : str, disk_path_tb_excluded : str, disk_path_for_cuts : str, out_file_name : str):
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
            
            # get wav language
            # wav_is_english = is_english_lang(xml_data)
            # if (not wav_is_english):
            #     print(f"File {wav_file_path} is not in English")
            
            # check for the length of the transcription
            #====================
            audiowav = AudioSegment.from_file(wav_full_path_current_disk)
            if (audiowav.duration_seconds > 30): # if the audio is longer than 30 seconds, cut
                print(f"File {wav_full_path_current_disk} is longer than 30s, processing cuts...")
                soup = BeautifulSoup(xml_data, "xml")
                for idx,segment in enumerate(soup.find_all("segment")):
                    start = float(segment.find("start").get_text())
                    end = float(segment.find("end").get_text())
                    # export the segment to the file
                    segment_name = f"{os.path.basename(wav_full_path_current_disk)}_segidx{idx}_{start}_{end}.wav"
                    segment_path = os.path.join(disk_path_for_cuts, segment_name)
                    audiowav[start*1000:end*1000].export(segment_path, format="wav")
                    
                    full_ts = get_fullts_for_one_segment(segment)
                    short_ts = get_shortts_for_one_segment(segment,wav_full_path_current_disk)
                    prompt_waypoints, prompt_short_callsigns, prompt_long_callsigns = get_prompt(xml_file)
                    
                    out_data.append({
                        "audio": segment_path.removeprefix(disk_path_tb_excluded).removeprefix('/'), # just want the path on the disk, not whole on the pc
                        "full_ts": full_ts,
                        "short_ts": short_ts,
                        "prompt": {
                        "waypoints": prompt_waypoints,
                        "short_callsigns": prompt_short_callsigns,
                        "long_callsigns": prompt_long_callsigns
                        }
                    })

                continue # dont need to add the original file
            #====================
            
            short_ts = get_shortts(wav_full_path_current_disk, xml_data)

            prompt_waypoints, prompt_short_callsigns, prompt_long_callsigns = get_prompt(xml_file)
            
            out_data.append({
                "audio": wav_file_path.removeprefix(disk_path_tb_excluded).removeprefix('/'), # just want the path on the disk, not whole on the pc
                "full_ts": full_ts,
                "short_ts": short_ts,
                "prompt": {
                    "waypoints": prompt_waypoints,
                    "short_callsigns": prompt_short_callsigns,
                    "long_callsigns": prompt_long_callsigns
                    }
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

def load_files_from_lists(split_list : str, ATCO_FOLDER_PATH) -> list:
    with open(split_list, 'r') as f:
        data = json.load(f)
        files_train = data['metadata_en_train.json']
        for file in files_train:
            file = os.path.join(ATCO_FOLDER_PATH, file)
        files_test_ruzyne = data['metadata_en_ruzyne_test.json']
        for file in files_test_ruzyne:
            file = os.path.join(ATCO_FOLDER_PATH, file)
        files_test_stefanik = data['metadata_en_stefanik_test.json']
        for file in files_test_stefanik:
            file = os.path.join(ATCO_FOLDER_PATH, file)
        files_test_zurich = data['metadata_en_zurich_test.json']
        for file in files_test_zurich:  
            file = os.path.join(ATCO_FOLDER_PATH, file)
        files_train_fr = data['metadata_fr_train.json']
        for file in files_train_fr:
            file = os.path.join(ATCO_FOLDER_PATH, file)
        files_test_fr = data['metadata_fr_test.json']
        for file in files_test_fr:
            file = os.path.join(ATCO_FOLDER_PATH, file)
        files_test_other = data['metadata_other_lang_test.json']
        for file in files_test_other:
            file = os.path.join(ATCO_FOLDER_PATH, file)
        
    return files_train, files_test_ruzyne,files_test_stefanik,files_test_zurich, files_train_fr, files_test_fr, files_test_other


if __name__ == "__main__":
    # SET ROOT DIR where ATCO2 FOLDER is stored ==============================================================
    # PATH TO ATCO FOLDER
    ROOT_DIR = '/run/media/johnny/31c5407a-2da6-4ef8-95ec-d294c1afec38/ATCO2-ASRdataset-v1_final/TESTING_ENV/'
    split_list = os.path.join("split_atco.json") # file with split files
    FOLDER_NAME = "ATCO2-ASRdataset-v1_final"  # CHANGE ONLY IF ATCO FOLDER NAME IS DIFFERENT
    # =============================================================
    ATCO_FOLDER_PATH = os.path.join(ROOT_DIR, FOLDER_NAME)
    files_train, files_test_ruzyne,files_test_stefanik,files_test_zurich, files_train_fr, files_test_fr, files_test_other = load_files_from_lists(split_list, ATCO_FOLDER_PATH)
    
    EN_disk_path_for_cuts = os.path.join(ATCO_FOLDER_PATH,"DATA-longer30s-cuts")
    os.makedirs(EN_disk_path_for_cuts, exist_ok=True)
    makemetadata(files_train, disk_path_tb_excluded=ROOT_DIR,disk_path_for_cuts=EN_disk_path_for_cuts, out_file_name="metadata_en_train.json")
    makemetadata(files_test_ruzyne, disk_path_tb_excluded=ROOT_DIR,disk_path_for_cuts=EN_disk_path_for_cuts,out_file_name="metadata_en_ruzyne_test.json")
    makemetadata(files_test_stefanik, disk_path_tb_excluded=ROOT_DIR,disk_path_for_cuts=EN_disk_path_for_cuts,out_file_name="metadata_en_stefanik_test.json")
    makemetadata(files_test_zurich, disk_path_tb_excluded=ROOT_DIR,disk_path_for_cuts=EN_disk_path_for_cuts,out_file_name="metadata_en_zurich_test.json")
    
    # for nonEN data
    NONEN_disk_path_for_cuts = os.path.join(ATCO_FOLDER_PATH,"DATA_nonEN-longer30s-cuts")
    os.makedirs(NONEN_disk_path_for_cuts, exist_ok=True)
    makemetadata(files_train_fr, disk_path_tb_excluded=ROOT_DIR,disk_path_for_cuts=NONEN_disk_path_for_cuts, out_file_name="metadata_fr_train.json")
    makemetadata(files_test_fr, disk_path_tb_excluded=ROOT_DIR,disk_path_for_cuts=NONEN_disk_path_for_cuts,out_file_name="metadata_fr_test.json")
    makemetadata(files_test_other, disk_path_tb_excluded=ROOT_DIR,disk_path_for_cuts=NONEN_disk_path_for_cuts,out_file_name="metadata_other_lang_test.json")
    
    