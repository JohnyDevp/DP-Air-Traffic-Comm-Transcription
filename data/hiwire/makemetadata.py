from calendar import c
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
    "sto": "00", "set":"00", "tisíc": "000",
    
    "decimal": ".", "point": ".", "dot": ".", "coma": ".",
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

def get_shortts(text, what : str ="alphanum",cutoff=None):
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

def makemetadata(audio_ts_doubles_dict : str, disk_path_tb_excluded : str, out_file_name : str):
    out_data = []
    
    for double in tqdm(audio_ts_doubles_dict):
        wav_full_path_current_disk = double['audio']
        full_ts = double['transcription']
 
        short_ts = get_shortts(full_ts, "alphanum")

        out_data.append({
                "audio": wav_full_path_current_disk.removeprefix(disk_path_tb_excluded).removeprefix('/'), # just want the path on the disk, not whole on the pc
                "full_ts": full_ts,
                "short_ts": short_ts,
                "prompt": None,
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

def split_wav_files(root : str) -> list:
    # counts for datasets ruzyne and stefanik - from each is used for test 350 files
    # and whole LZSH_Zurich is used for test
    test_dirs_sp = ["SJJM","SLGF","SMCM"]
    test_dirs_fr = ["FMHF","FMKF","FOMF","FSVF","FSHM","FSPM","FTEM","FVCM"]
    test_dirs_gr = ["GTAM","GVPM","GVSF","GNSF"]
    
    result_train = []
    result_test_fr = []
    result_test_gr = []
    result_test_sp = []
    main_file = os.path.join(root, "hiwire.mlf")
    with open(main_file, "r") as f:
        lines = f.readlines()
        pattern = r'\".*[.]lab\"'
        
        current_file = ""
        current_file_transcription = []
        
        def append_wav_to_result(current_file, current_file_transcription):
            if (current_file != ""):
                # check for file existence
                if (not os.path.exists(current_file)):
                    print(f"File {current_file} does not exist, skipping")
                    return
                file_spec = os.path.basename(current_file).split('_')[0]
                if file_spec in test_dirs_fr:
                    result_test_fr.append({"audio":current_file, "transcription": ' '.join(current_file_transcription)})
                elif file_spec in test_dirs_gr:
                    result_test_gr.append({"audio":current_file, "transcription": ' '.join(current_file_transcription)})
                elif file_spec in test_dirs_sp:
                    result_test_sp.append({"audio":current_file, "transcription": ' '.join(current_file_transcription)})
                else:
                    result_train.append({"audio":current_file, "transcription": ' '.join(current_file_transcription)})
            
        for line in lines:
            if (re.match(pattern, line.strip())):
                if (current_file != ""):
                    append_wav_to_result(current_file, current_file_transcription)
                    current_file = ""
                    current_file_transcription = []
                    
                # build the wav file path
                current_file = line.replace('"','').strip().removesuffix(".lab").removeprefix("*/") + '_LN.wav'
                # possible dirs
                dirs = ['FR','GR','SP']
                mezzo_path = 'speechdata/LN'
                file_spec = current_file.split('_')[0]
                for dir in dirs: # check for the first letter to decide the directory
                    if (file_spec[0].__eq__(dir[0])):
                        current_file = os.path.join(root, mezzo_path, dir, file_spec, current_file)
                        
                        break
            elif (line.strip() == "."):
                append_wav_to_result(current_file, current_file_transcription)
                current_file = ""
                current_file_transcription = []
            else: 
                if (current_file != ""):
                    current_file_transcription.append(line.strip())
        
        # append the last possible processed file
        if (current_file != ""):
            append_wav_to_result(current_file, current_file_transcription)
            current_file = ""
            current_file_transcription = []
    
    return result_train, result_test_fr, result_test_gr, result_test_sp
    
if __name__ == "__main__":
    ROOT_DIR="/run/media/johnny/31c5407a-2da6-4ef8-95ec-d294c1afec38/HIWIRE_ELDA_S0293/"
    
    files_train, files_test_fr, files_test_gr, files_test_sp = split_wav_files(ROOT_DIR)
 
    # files_train, files_test_ruzyne,files_test_stefanik,files_test_zurich = split_wav_files(ROOT_DIR_EN)
    makemetadata(files_train, disk_path_tb_excluded="/run/media/johnny/31c5407a-2da6-4ef8-95ec-d294c1afec38", out_file_name="metadata_hwir_train.json")
    makemetadata(files_test_fr, disk_path_tb_excluded="/run/media/johnny/31c5407a-2da6-4ef8-95ec-d294c1afec38",out_file_name="metadata_hwir_fr_test.json")
    makemetadata(files_test_gr, disk_path_tb_excluded="/run/media/johnny/31c5407a-2da6-4ef8-95ec-d294c1afec38",out_file_name="metadata_hwir_gr_test.json")
    makemetadata(files_test_sp, disk_path_tb_excluded="/run/media/johnny/31c5407a-2da6-4ef8-95ec-d294c1afec38",out_file_name="metadata_hwir_sp_test.json")

   
    