import re

from rapidfuzz import process
from pydub import AudioSegment 
import os, json, time

import tqdm

from data.DPout.data.hiwire.makemetadata import FOLDER_NAME

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
    
    "decimal": ".", "point": ".", "dot":"."
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

digit_reverse_map = {
                     "0": "zero", "1": "one", "2": "two", "3": "three",
                     "4": "four", "5": "five", "6": "six", "7": "seven",
                     "8": "eight", "9": "nine",
                     
                     "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen",
                     "14": "fourteen", "15": "fifteen", "16": "sixteen",
                     "17": "seventeen", "18": "eighteen", "19": "nineteen",
                     
                     "20": "twenty", "30": "thirty", "40": "forty",
                     "50": "fifty", "60": "sixty", "70": "seventy",
                     "80": "eighty", "90": "ninety"
                     }

unit_reverse_map = {
    "0": "zero", "1": "one", "2": "two", "3": "three",
                     "4": "four", "5": "five", "6": "six", "7": "seven",
                     "8": "eight", "9": "nine",
}
tens_reverse_map = {
    "20": "twenty", "30": "thirty", "40": "forty",
                     "50": "fifty", "60": "sixty", "70": "seventy",
                     "80": "eighty", "90": "ninety"
}
teens_reverse_map = {
    "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen",
                     "14": "fourteen", "15": "fifteen", "16": "sixteen",
                     "17": "seventeen", "18": "eighteen", "19": "nineteen"
}

aviation_reverse_map = {
    "A": "alpha", "B": "bravo", "C": "charlie", "D": "delta",
    "E": "echo", "F": "foxtrot", "G": "golf", "H": "hotel",
    "I": "india", "J": "juliet", "K": "kilo", "L": "lima",
    "M": "mike", "N": "november", "O": "oscar", "P": "papa",
    "Q": "quebec", "R": "romeo", "S": "sierra", "T": "tango",
    "U": "uniform", "V": "victor", "W": "whiskey",
    "X": "x-ray", "Y": "yankee", "Z": "zulu"
}

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
    
def get_shortts(full_ts : str, what : str ="alphanum"):
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
    
    # remove everything between [ ], if present
    reg = re.compile(r"\[[^\]]*\]")
    full_ts = reg.sub("", full_ts)
    
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

def alphaToAviation(ts : str) -> str:
    result = []
    for word in ts.split(" "):
        word_upper = word.upper()
        if word_upper in aviation_reverse_map:
            result.append(aviation_reverse_map[word_upper])
        else:
            result.append(word)
            
    return " ".join(result)

def complexNumberToWords(number : str) -> str:
    # we assume the number is kind 1523 ... less or more
    #! we assume, that numbers spoken as digits were separated by space and so are converted to digits and not get there
    # we KNOW the max length of the number is 4
    result = []
    for i in range(len(number)-2, -1, -1):
        if i == len(number)-2: # tens and units
            if number[i] == "0" and number[i+1] == "0": # units and tens are zeros , 00, SKIP
                continue
            elif number[i] == "0" and number[i+1] != "0": # meaning the units stands alone .. 01, 02, 03, ...
                result.insert(0,unit_reverse_map[number[i+1]])
            elif number[i] != "0" and number[i+1] == "0": # 10,20,30,....
                result.insert(0,tens_reverse_map[number[i] + "0"])
            elif number[i] != "0" and number[i+1] != "0":
                if number[i] == "1":
                    result.insert(0,teens_reverse_map[number[i] + number[i+1]])
                else:
                    result.insert(0,unit_reverse_map[number[i+1]])
                    result.insert(0,tens_reverse_map[number[i] + "0"])
        elif i == len(number)-3: # hundreds
            if number[i] != "0":
                result.insert(0,"hundred")
                result.insert(0,unit_reverse_map[number[i]])
        elif i == len(number)-4: # thousands and ten-thousands
            #! beware, i am using -1, because in the first if statement i look backward (according to the i (index)), but now i am looking for potential next number
            #! also according to the, which means i need to go -1
            if i-1 == 0: # 11000, etc.
                if number[i-1] != "0" and number[i] == "0": # 10 000,20 000,30 000,....
                    result.insert(0,tens_reverse_map[number[i-1] + "0"])
                elif number[i-1] != "0" and number[i] != "0": # 11 000, 12 000, 13 000, 23 000, 43 000...
                    result.insert(0,"thousand")
                    if number[i-1] == "1":
                        result.insert(0,teens_reverse_map[number[i-1] + number[i]])
                    else:
                        result.insert(0,unit_reverse_map[number[i]])
                        result.insert(0,tens_reverse_map[number[i-1] + "0"])
                    
            else:
                result.insert(0,"thousand")
                result.insert(0,unit_reverse_map[number[i]]) # 1000, 2000, 3000, ...
    
    return ' '.join(result)

def numbersToWords(ts : str) -> str:
    result = []
    for word in ts.split(" "):
        # either the number is right in the reverse map
        if word in digit_reverse_map:
            result.append(digit_reverse_map[word])
        elif word.isnumeric(): # this IF need to be before the next one, because the next can catch these
            number_as_words = complexNumberToWords(word).split(" ")
            result.extend(number_as_words)
        elif word.replace(".", "").isnumeric():
            # if the number is float
            for digit in word:
                if digit == ".":
                    result.append("point")
                    continue
                result.append(digit_reverse_map[digit])
        else:
            result.append(word)
            
    return " ".join(result)     

def getFullTs(ts) -> None|str:
    
    ts = ts.strip()
    # check whether it is not blank
    if ts == "" or ts == "..":
        return None
    
    
    # remove plus signs
    ts = re.sub(r'\+', '', ts)
    # remove the tags
    ts = re.sub(r'\[[^\]]*\]', ' ', ts)
    # parse the pronountiation (e.g. "(9(najn))" to '9')
    ts = re.sub(r'\(([^\)]+)\([^\)]+\)\)', r' \1 ', ts)
    # remove double spaces
    ts = re.sub(r'\s+', ' ', ts)
    
    # put floats together
    ts = re.sub(r'(?<=\d)\s+\.\s+(?=\d)', '.', ts) # 1 . 2 -> 1.2
    
    # now parse all the numbers back to spoken form
    ts = numbersToWords(ts)
    ts = alphaToAviation(ts)
    # replace FL for flight level
    ts = ts.replace("FL", "flight level")
    
    return ts.strip()

def make_audio_split(start_in_s,stop_in_s,audio,out_path) -> str:
    segment = audio[start_in_s*1000:stop_in_s*1000]
    segment.export(out_path, format="wav")

def makeMetadata(parsed_wavs, path_to_wavs, out_dir_wavs, out_path_metadata_train, out_path_metadata_test, out_dir_replace = ""):
    test_wavs = []
    with open('test_wavs.out', "r") as f:
        for line in f.readlines():
            test_wavs.append(line.split()[0].strip())
    
    result_test_set = []
    result_train_set = []
    
    for wav in tqdm.tqdm(parsed_wavs):
        wavname = wav["audio"]
        start = float(wav["start"])
        end = float(wav["end"])
        full_ts = wav["full_ts"]
        short_ts = wav["short_ts"]
        
        # load proper audio file
        audio = AudioSegment.from_wav(os.path.join(path_to_wavs, wavname) + ".wav")

        # extract the specified part of the audio
        audio_out_path = os.path.join(out_dir_wavs, wavname)
        suffix = 0
        while os.path.exists(audio_out_path + f"_{suffix}.wav"):
            suffix += 1
        audio_out_path = audio_out_path + f"_{suffix}.wav"
        make_audio_split(start, end, audio, audio_out_path)
        
        # save audio metadata to proper set
        if wavname in test_wavs:
            result_test_set.append({
                "audio": audio_out_path.replace(out_dir_replace, ""),
                "full_ts": full_ts,
                "short_ts": short_ts,
                "prompt": None
            })
        else:
            result_train_set.append({
                "audio": audio_out_path.replace(out_dir_replace, ""),
                "full_ts": full_ts,
                "short_ts": short_ts,
                "prompt": None
            })
    
    if  os.path.exists(out_path_metadata_train):
        out_path_metadata_train.replace(".json", f"_{time.time()}.json")
    if  os.path.exists(out_path_metadata_test):
        out_path_metadata_test.replace(".json", f"_{time.time()}.json")
        
    with open(out_path_metadata_train, "w") as f:
        f.write(json.dumps(result_train_set, indent=4))
    
    with open(out_path_metadata_test, "w") as f:
        f.write(json.dumps(result_test_set, indent=4))
        
def parseStm(path):
    result = []
    skipped = 0
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.split('<,,,>')
            wav_desc = line[0].split(" ")
            ts = line[1] # this is unprocessed transcription as is in the stm file
            wavname = wav_desc[0]
            wav_start_time = wav_desc[3]
            wav_end_time = wav_desc[4]
            
            if float(wav_end_time) - float(wav_start_time) >= 30:
                print(f"Skipping {wavname} because it is longer than 30s")
                skipped += 1
                continue
                
            full_ts = getFullTs(ts)
            if full_ts is None or full_ts.strip() == "":
                skipped += 1
                continue
            
            short_ts = get_shortts(full_ts)

            result.append({
                "audio": wavname,
                "start": wav_start_time,
                "end": wav_end_time,
                "full_ts": full_ts,
                "short_ts": short_ts
            })

    print(f"WARNING: Skipped {skipped} files")
    return result

if __name__ == "__main__":
    
    #======================================================
    # CHANGE THIS TO YOUR DISK PATH
    DISK_ROOT = '/run/media/johnny/31c5407a-2da6-4ef8-95ec-d294c1afec38'
    #======================================================
    # IT IS POSSIBLE THAT THESE PATH WILL NEED TO BE CHANGED
    FOLDER_NAME = 'UWB_ATCC'
    path_to_wavs = f'{DISK_ROOT}/UWB_ATCC/audio'
    
    # there will be stored audio splits, which are made during running of this script
    # WARNING: this folder should exist and be empty
    path_to_save_new_wavs = f'{DISK_ROOT}/UWB_ATCC/audio_split' 
    os.makedirs(path_to_save_new_wavs, exist_ok=True)
    replace_path_of_audio = f'{DISK_ROOT}' # this part won't be stored in the metadata
    out_metadata_test = './metadata_test.json'
    out_metadata_train = './metadata_train.json'
    
    parsed_wavs = parseStm('/run/media/johnny/31c5407a-2da6-4ef8-95ec-d294c1afec38/UWB_ATCC/stm/stm')
    
    makeMetadata(parsed_wavs, 
                 path_to_wavs, 
                 path_to_save_new_wavs,
                 out_metadata_train,
                 out_metadata_test,
                 replace_path_of_audio)
    
   