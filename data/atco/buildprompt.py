from math import e
from operator import call
import json,os
from sys import path
from bs4 import BeautifulSoup
import re
from unicodedata import normalize
import random
from tqdm import tqdm
from rapidfuzz import process

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
    "sto": "00", "set":"00", "tisíc": "000", 'tisíce': "000",
    
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

# ========================================================================================================
vocab_callsign = {}
vocab_airline = {}
def _key_normalizer(key):
    key = str(key).lower().strip() # build something like airforceonetwotree
    key = re.sub(r'\s+','',key) # remove all spaces
    return key

# make vocabularies, that will be used for obtaining the ICAO codes from the callsigns
def make_vocab(js_callsign_icao, js_airline_icao):
    for callsign in js_callsign_icao:
        vocab_callsign[_key_normalizer(callsign)] = js_callsign_icao[callsign]['icao']
    
    for airline in js_airline_icao:
        vocab_airline[_key_normalizer(airline)] = js_airline_icao[airline]['icao']

info_vocab = {}
def make_info_vocab(info_file_path):
    with open(info_file_path, "r") as f:
        lines = f.readlines()[8:] # first 8 lines are not needed
        for line in lines:
            line_split = line.split(':')
            if (len(line_split) != 2 or line_split[0].strip().__len__() < 4 or line_split[1].split(' ').__len__() > 10):
                continue
            code = line.split(":")[0].lower().strip()
            full = line.split(":")[1].lower().strip()
            info_vocab[_key_normalizer(full)] = code
    
    return info_vocab 

# ========================================================================================================      
def run_xmlfile_process(xml_data, info_file_path):
    soup = BeautifulSoup(xml_data, "xml")
    out = {}
    for segment in soup.find_all("segment"):
        found_tags = get_knowledge(segment, info_file_path)
        
        # merge the dictionaries of found parts
        for tag, content in found_tags.items():
            if tag in out:
                out[tag].extend(content)
            else:
                out[tag] = content
    return out

def get_knowledge(segment, info_file_path) -> str:
    # first obtain tags from the text of the segment
    text = segment.find("text").text
    # remove multiply tagged one, logically together belonging content (meaning one tag)
    text=re.sub(r'\[\/#(\w+)\]\s*\[#\1\]',' ',text)
    # remove multiple spaces
    text = re.sub(r'\s+', ' ',text)
    
    out = {}
    
    # CALLSIGN TAG PROCESS
    # extract the speaker label
    speaker_label = segment.find("speaker_label").text if not ('UNK'.lower() in segment.find("speaker_label").text.lower()) else None
    cal_out = get_callsigns_from_text(text, info_file_path)
    out['long_callsigns'] = cal_out['long']
    if (speaker_label):
        out['short_callsigns'] = [speaker_label]
    else: out['short_callsigns'] = cal_out['short']
    
    # RUNWAY
    pass 
    # TAXIWAY
    pass

    return out

def get_callsigns_from_text(corrected_text_with_tags, info_file_path):
    out = {'short' : [],'long' : []}
    
    # find all callsigns
    pattern = r'\[#callsign\](.*?)\[/#callsign\]'
    for it in re.findall(pattern, corrected_text_with_tags):
        # append the found callsign to the dictionary, remove multiple spaces and normalize the string (hex characters)
        callsign = re.sub(r'\s+', ' ', normalize('NFC',it))
        out['long'].append(callsign)
        
        # shorten the callsign
        full_vocab = make_info_vocab(info_file_path)
        if (_key_normalizer(callsign) in full_vocab):
            out['short'].append(full_vocab[_key_normalizer(callsign)].upper())
        else:
            # it is not found in the dictionary, so we will process it in shortennign function
            out['short'].append(process_callsign_with_vocabs(callsign))

    return out

def process_callsign_with_vocabs(callsign : str):
    # this function works for callsigns like Air France 1234, where the last word of digits
    # we obtain by calling process_tag_content, and Air France we translate to shortcut by 
    # looking into the info file
    
    partly_processed_callsign = process_tag_content(callsign, "alphanum")
    # we assume that the last word is now built from numbers and letters recognized 
    # from air traffic alphabet and numbers
    
    # we will try to find the airport code in the info file    
    airport_name = ' '.join(partly_processed_callsign.strip().lower().split(' ')[0:-1]).strip()
    airport_name_norm = _key_normalizer(airport_name)
    
    if (airport_name_norm.strip() != ""):
        if (airport_name_norm in vocab_airline):
            airport_code = vocab_airline[airport_name_norm]
            return (airport_code + partly_processed_callsign.split(' ')[-1]).replace(' ','').upper()
        elif (airport_name_norm in vocab_callsign):
            airport_code = vocab_callsign[airport_name_norm]
            return (airport_code + partly_processed_callsign.split(' ')[-1]).replace(' ','').upper()
    
    # if we cant shorten whole callsign return it as it is
    return partly_processed_callsign.strip()

def sample_random_callsigns(set_of_callsigns, n, exclude = []):
    return random.sample([x for x in set_of_callsigns if x not in exclude], min(n,len(set_of_callsigns)))
    

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
    
    # create vocab
    make_vocab(json.load(open('../tools/callsigns_icao.json')), json.load(open('../tools/airline_icao.json')))
    for file in METADATA_PATHS:
        with open(file,'r') as f:
            js_file = json.load(f)
            for meta in tqdm(js_file):
                # extract the recording file name without extension
                rec_file = os.path.basename(meta['audio']).split('.wav')[0]
                # build up the correct path to the current disk, where the file should be located
                for fl in FOLDERS_TO_SEARCH:
                    p = os.path.join(DISK_PATH,fl,rec_file+'.wav')
                    if os.path.exists(p):
                        path = p
                        break
                if not path: continue
                
                # open xml and info files
                xml_file = path.replace('.wav','.xml')
                info_file_path = path.replace('.wav','.info')
                
                # from xml file obtain all callsigns
                with open(xml_file,'r') as f:
                    xml_data = f.read()
                    out = run_xmlfile_process(xml_data, info_file_path)
                
                meta['prompt-data'] = {
                    'long_callsigns': list(set(out['long_callsigns'])),
                    'short_callsigns': list(set(out['short_callsigns'])),
                    'waypoints': meta['prompt']['waypoints'],
                    'nearby_short_callsigns': meta['prompt']['short_callsigns'],
                    'nearby_long_callsigns': meta['prompt']['long_callsigns'],
                    'short_runway': [],
                    'long_runway': [],
                    'short_taxiway': [],
                    'long_taxiway': []
                }
                len_waypoints = len(meta['prompt-data']['waypoints'])
                len_long_callsign = len(meta['prompt-data']['long_callsigns'])
                
                # build some prompts, that can be used during training
                # this prompt is using 1 correct callsign and 4 incorrect callsigns
                meta['prompt_fullts_1G_4B'] = ', '.join(meta['prompt-data']['long_callsigns']) + ', ' + \
                    ', '.join(sample_random_callsigns(meta['prompt-data']['nearby_long_callsigns'],4))
                # this prompt uses 5 incorrect callsigns
                meta['prompt_fullts_5B'] = ', '.join(sample_random_callsigns(meta['prompt-data']['nearby_long_callsigns'],5))
                meta['prompt_fullts_50B'] = ', '.join(sample_random_callsigns(meta['prompt-data']['nearby_long_callsigns'],50))
                
                # remove the prompt from the metadata (because in the old version it was there with the content sorted above)
                meta.pop('prompt')
            
            json.dump(js_file, open(SAVE_FOLDER+file,'w'),indent=4)
            f.close()