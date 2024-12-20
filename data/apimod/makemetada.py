import os 
import glob as glob
from rapidfuzz import process
from tqdm import tqdm
import json, re
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# set up the transcribing environment
import torch
from transformers import pipeline

# load the model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
transcribe = pipeline(task="automatic-speech-recognition", model="BUT-FIT/whisper-ATC-czech-full", chunk_length_s=30, device=device)
transcribe.generation_config.language = "english"
transcribe.generation_config.task = "transcribe"

def make_transcription(file_path):
    # read the audio file
    transcription = transcribe(file_path)
    return transcription['text']

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
    result = re.sub(r'\s+([.,!?])', r'\1', result) # remove spaces before punctuation
    return result

def airtraffic_transcript_to_code_(transcript : str):
    result = ""
    for line in transcript.splitlines():
        result += get_shortts(line, "alphanum")
        
        # Append a newline character for each line
        result += "\n"
                
    return result.strip()

def do_the_transcriptions(DIR_WITH_WAVS, SAVE_FILE_PATH):
    metadata = []
    for file_path in tqdm(glob.glob(DIR_WITH_WAVS+"/*.wav")):
        transcription = make_transcription(file_path)
        metadata.append(
            {
                "audio":file_path,
                "full_ts": transcription,
                "short_ts":airtraffic_transcript_to_code_(transcription),
                "prompt": None,
            }
        )

    # save the metadata, ensure no overwriting of previously created metadata
    out=json.dumps(metadata,indent=4,ensure_ascii=False)
    name = SAVE_FILE_PATH
    if (os.path.exists(name)):
        print(f"{SAVE_FILE_PATH} already exists, creating a new one with a timestamp")
        name = f"{SAVE_FILE_PATH}{time.time()}.json"
        
    with open(SAVE_FILE_PATH,"w") as f:
        f.write(out)

if __name__ == "__main__":
    DIRS="/run/media/johnny/31c5407a-2da6-4ef8-95ec-d294c1afec38/A-PiMod/2013_10_Christoph/01_02_EL_LN_UN_VV_YADA"
    SAVE_FILE_PATH="./metadata_train.json"
    do_the_transcriptions(DIRS,SAVE_FILE_PATH)
    