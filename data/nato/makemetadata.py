# {
#   "filepath":str,
#   "full_ts":str,
#   "short_ts":str,
#   "language":str     
# }

import glob as glob
import os
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

def contains_numbers_and_letters(text):
    has_letter = any(char.isalpha() for char in text)
    has_number = any(char.isdigit() for char in text)
    return has_letter and has_number

# Replace specific tags and their content
def replace_tag_with_word(text, tag, replacement):
    pattern = fr"<{tag}.*?>.*?</{tag}>"
    return re.sub(pattern, replacement, text)

def airtraffic_transcript_to_code_(transcript : str):
    # Combine both maps
    combined_map = {**number_map, **aviation_map}

    # Process the words
    result = []
    current_transcript = ""

    for line in transcript.splitlines():
        current_transcript = ""
        for word in line.split():
            # Find the closest match from the combined_map keys
            bestmatch = process.extractOne(word.lower(), combined_map.keys(), score_cutoff=93)
            # print(bestmatch)
            if bestmatch and word.lower() not in leave_untouch_words:
                current_transcript += combined_map[bestmatch[0]]  # Add the matched transcription
            else:
                if current_transcript:  # If there is an ongoing transcription chunk
                    result.append(current_transcript)
                    current_transcript = ""
                result.append(word)  # Append the normal word as-is
        
        
        # Append any leftover transcription chunk
        if current_transcript:
            result.append(current_transcript)
            
        # Append a newline character for each line
        result.append("\n") 
                

    # Join the result with spaces to maintain the sentence structure
    return ' '.join(result)

def process_tag_content(text, what : str ="alphanum"):
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

def get_shortts_from_tra(fullpath_tra : str):
    with open(fullpath_tra, "r") as f:
        all_text = f.read()

        # check if cmd exists - if so, obtain the CALLSIGN from it
        cmd_path = fullpath_tra.replace(".tra", ".cmd")
        if os.path.exists(cmd_path):    
            with open(cmd_path, "r") as f:
                callsign = f.readline().split(' ')[0]
                
                # check whether the callsign is really a callsign
                if (not contains_numbers_and_letters(callsign)):
                    callsign = ""
        else:
            callsign = ""
        
        tags_tobe_replaced = ["degree_absolute", "runway", "altitude", "speed", "distance"]
        # replace the callsign in the text
        if callsign:
            all_text = replace_tag_with_word(all_text, "callsign", callsign)
        else:
            tags_tobe_replaced.append("callsign")
            
        # list of tags to be replaced 
        for tag in tags_tobe_replaced:
            # Create the regex pattern for the current tag
            pattern = rf"<{tag}.*?>(.*?)</{tag}>"
            
            # Define the replacement function
            def replace_match(match):
                # Extract the content inside the tag
                tag_content = match.group(1)
                # Process the content using the function
                if tag == "callsign":
                    processed_content = process_tag_content(tag_content, "alphanum")
                else:
                    processed_content = process_tag_content(tag_content, "num")
                    
                # Replace the tag and its content with the processed content
                return processed_content
            
            # Use re.sub to replace all matches for the current tag
            all_text = re.sub(pattern, replace_match, all_text)
        
        # after all remove the rest of tags, that hasnt been replaced
        # Parse the text
        soup = BeautifulSoup(all_text, "html.parser")
        # Extract plain text
        plain_text = soup.get_text(separator=" ", strip=True)
        
        return plain_text
        
def get_shortts(wav_full_path_current_disk, fullts) -> str:
    # get short ts
    
    # check whether the .tra file exists
    tra_path = wav_full_path_current_disk.replace(".wav", ".tra")
    if (os.path.exists(tra_path)):
        # 1. check whether .tra - if it does, check for the .cmd file with callsign
        shortenedts= get_shortts_from_tra(tra_path)
        # still pass it to my function for short ts, because some numbers might not be tagged
        return airtraffic_transcript_to_code_(shortenedts)
    else:
        # 2. if it does not, use the fullts shortened by function
        return airtraffic_transcript_to_code_(fullts)

def get_fullts(wav_full_path_current_disk) -> str | Exception:
    # get the full ts from the .cor file, otherwise raise an exception
    cor_file = wav_full_path_current_disk.replace(".wav", ".cor")
    if (os.path.exists(cor_file)):
        with open(cor_file, "r") as f:
            return f.read()
    else:
        raise Exception(f"File {cor_file} does not exist")

def makemetadata(PATH_TO_LISTINGS_OF_FILES, DISK_PATH, SAVE_PATH):
    out_data = []
    with open(PATH_TO_LISTINGS_OF_FILES, "r") as f:
        for line in tqdm(f.readlines()):
            # append path for the wavfile
            #! path prefix leading to root of my disk, where the data are stored, to LOWWXX, from where the path is unique for the file
            #! this is done because the path in the wav.scp is wrong
            path_prefix = "MALORCA/DATA_ATC/VIENNA/WAV_FILES"
            wav_file_path=os.path.join(path_prefix, line.split('/VIENNA/')[1].strip())
            wav_full_path_current_disk = os.path.join(DISK_PATH, wav_file_path)               
            
            # if the full ts is not available, skip the file
            try:
                fullts = get_fullts(wav_full_path_current_disk)
                shortts = get_shortts(wav_full_path_current_disk, fullts)
            except Exception as e:
                print(e)
                continue
            
            out_data.append({
                "audio": wav_file_path,
                "full_ts": fullts,
                "short_ts": shortts,
                "prompt":None,
            })
                
    # save the metadata  
    # choose file name
    out=json.dumps(out_data,indent=4,ensure_ascii=False)
    if (os.path.exists(SAVE_PATH)):
        name = f"{SAVE_PATH}{time.time()}.json"
    else:
        name = SAVE_PATH
    
    # save the file
    with open(name,"w") as f:
        f.write(out)
    
if __name__ == "__main__":
    # PATH_TO_LISTINGS_OF_FILES = "/run/media/johnny/31c5407a-2da6-4ef8-95ec-d294c1afec38/MALORCA/DATA_ATC/VIENNA/DATA/test/wav.scp"
    # DISK_PATH="/run/media/johnny/31c5407a-2da6-4ef8-95ec-d294c1afec38"
    # SAVE_PATH="./metadata_test.json"
    
    # makemetadata(PATH_TO_LISTINGS_OF_FILES, DISK_PATH, SAVE_PATH)
    