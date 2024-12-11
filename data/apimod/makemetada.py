import os 
import glob as glob
from rapidfuzz import process
from tqdm import tqdm
import json
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
    
    "hundred": "00", "thousand": "000",
    "hundert": "00", "tausend": "000",
    "sto": "00", "tisíc": "000"
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

def airtraffic_transcript_to_code_(transcript : str, number_map, aviation_alphabet_map):

    # Combine both maps
    combined_map = {**number_map, **aviation_alphabet_map}

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

def do_the_transcriptions(DIR_WITH_WAVS, SAVE_FILE_PATH):
    metadata = []
    for file_path in tqdm(glob.glob(DIR_WITH_WAVS+"/*.wav")):
        transcription = make_transcription(file_path)
        metadata.append(
            {
                "audio":file_path,
                "full_ts": transcription,
                "short_ts": airtraffic_transcript_to_code_(transcription,number_map,aviation_map),
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