import os
from pydub import AudioSegment 
import re
import json
from rapidfuzz import process
from glob import glob
from time import time
from tqdm import tqdm


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

def extract_short_segments(ts_file, audio_in, wav_out_path, max_duration=30):
    """
    Extracts segments from the input file where the duration between <Sync> tags is less than max_duration.
    
    :param ts_file: Path to the input file.
    :param output_file: Path to save the filtered content.
    :param max_duration: Maximum duration in seconds between <Sync> tags to retain content.
    """
    audio = AudioSegment.from_file(audio_in)
    
    with open(ts_file, "r", encoding="utf-8",errors='ignore') as file:
        content = file.read()
    
    # Regex to match <Sync time="..."/> and capture timestamps
    sync_pattern = r'<Sync time="([\d.]+)"/>'
    
    # Extract text between Sync tags
    split_content = re.split(sync_pattern, content)
    text_segments = split_content[1:-1]  # Skip the first and last split as it precedes and lasts the first <Sync>
    
    # extract all speakers 
    speakers_dict = {}
    pattern = r'<Speaker\s+[^>]*id="(.*?)"\s+[^>]*name="(.*?)"(?:\s+[^>]*type="(.*?)")?'
    suma = 0
    for match in re.finditer(pattern, split_content[0]):
        speakers_dict[match.group(1)] = {
            "sign":match.group(2).split('_')[-1],
            "type":match.group(3)
        }
        
    short_segments = []
    
    #extract the very first speaker (from split_content[0]) and set the "last_speaker"
    first_speaker = re.findall(r'speaker="(.*?)"', split_content[0])
    if (first_speaker == []):
        current_speaker = None
    else:
        current_speaker = first_speaker[-1]
        
    # Process pairs of Sync timestamps and their corresponding text
    audio_files_counter = 0
    for i in range(0, len(text_segments),2):
        if (i+2) >= len(text_segments):
            break
        
        current_time = float(text_segments[i])
        next_time = float(text_segments[i + 2])
        text = text_segments[i + 1]
        
         # get speaker for the next segment
        next_speaker = re.findall(r'speaker="(.*?)"', text)
        if (next_speaker != []):
            next_speaker = next_speaker[-1]
        else:
            next_speaker = current_speaker
            
        # if the duration is lower than 30s (max input for whisper)
        duration = next_time - current_time
        if duration < max_duration:
            # remove all tags from the text
            text = re.sub(r'<(?!Turn\b)[^>]*>', '', text).strip()
            text = re.sub(r'\n+', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            # now i have left only Turn tags, each is meaning, that there is another speaker, so put a new line on that places
            # and strip then - beacuse there can be a new line at the end or beggining of the text
            text = re.sub(r'<*Turn[^>]*>', '\n', text).strip()
            # and remove trailing dots from the end
            text = re.sub(r'(\s*\.\s*)+$', '', text)
            # and also from the begginning
            text = re.sub(r'^(\s*\.\s*)+', '', text)
            
            speaker_sign = speakers_dict[current_speaker]["sign"] if speakers_dict.get(current_speaker) else "unk"
            gender = speakers_dict[current_speaker]["type"] if speakers_dict.get(current_speaker) else "unk"
            speech_audio_file = get_audio_split(current_time,next_time,audio,wav_out_path,f"{audio_files_counter}_{speaker_sign}_{gender}.wav")
            # add segment only if it is not empty
            if (text != ""):
                short_segments.append({
                    "start": current_time,
                    "end": next_time,
                    "audio": speech_audio_file,
                    "speakers": speaker_sign,
                    "type": gender,
                    "text": text
                })
        
        # set the last speaker for the next segment
        current_speaker = next_speaker
        
    return short_segments

def get_audio_split(start_in_s,stop_in_s,audio,outdir,name:str) -> str:
    segment = audio[start_in_s*1000:stop_in_s*1000]
    i = 0
    newname = name
    while os.path.exists(os.path.join(outdir,newname)):
        newname = name.removesuffix('.wav') + '_' + str(i) + '.wav'
        i += 1
    segment.export(os.path.join(outdir,newname), format="wav")
    return os.path.join(outdir,newname)
    
def make_metadata(short_segments,set_lang,out_file_path="metadata.json"):
    test_set_uk = ['L0H','H7V','O6N']    
    test_set_ca = ['CB','RT','AW','M2D']
    test_set_nl = ['AMAA','31','PAVO','PAFF']
    test_set_de = ['2EZ','B6J','Y4B','A2W','7KD','V1Q','L2F']
    
    metadata_test = []
    metadata_train = []
    
    match set_lang:
        case "UK":
            test_set = test_set_uk
        case "CA":
            test_set = test_set_ca
        case "NL":
            test_set = test_set_nl
        case "DE":
            test_set = test_set_de

    tests_written = 0
    train_written = 0
    
    for segment in tqdm(short_segments):
        audio_path = segment["audio"].replace("/run/media/johnny/31c5407a-2da6-4ef8-95ec-d294c1afec38/","")
        full_ts = segment["text"]
        short_ts = get_shortts(full_ts)
        
        if (segment["speakers"] in test_set):
            metadata_test.append({
                "audio":audio_path,
                "full_ts":full_ts,
                "short_ts":short_ts,
                "prompt": None
            })
            tests_written += 1
        else:
            metadata_train.append({
                "audio":audio_path,
                "full_ts":full_ts,
                "short_ts":short_ts,
                "prompt": None
            })
            train_written += 1
    
    test_file_path = out_file_path.replace(".json","_test.json")
    train_file_path = out_file_path.replace(".json","_train.json")
    if (os.path.exists(out_file_path)):
        bsname = os.path.basename(out_file_path).removesuffix(".json")
        out_file_path = out_file_path.replace(bsname,bsname+str(time()))
    with open(test_file_path, "w") as file:
        json.dump(metadata_test, file, indent=4)
    with open(train_file_path, "w") as file:
        json.dump(metadata_train, file, indent=4)
    
    print(f"** SET: {set_lang} **")
    print(f"Test set: {tests_written} segments;", f"Train set: {train_written} segments")

def speakers_split(short_segments):
    speaker_groups = []
    suma = 0
    recordings_length = 0
    for segment in short_segments:
        speakers = segment["speakers"]
        found = False
        for gr in speaker_groups:
            if (gr['speakers'] == speakers):
                gr['count'] += 1
                found = True
                break
        if (not found):
            speaker_groups.append({
                "speakers": speakers,
                "type": segment["type"],
                "count": 1
            })
        recordings_length += segment["end"]-segment["start"]
        suma += 1
    print("SUMA:",suma, " RECORDINGS LENGTH:", recordings_length / 60)
    print(speaker_groups)
        
if __name__ == "__main__":
    DISK_ROOT = "/run/media/johnny/31c5407a-2da6-4ef8-95ec-d294c1afec38/"
    inputs = [
        DISK_ROOT + "n4_nato_speech_LDC2006S13/data/UK/UK_",
        DISK_ROOT + "n4_nato_speech_LDC2006S13/data/CA/CA_",
        DISK_ROOT + "n4_nato_speech_LDC2006S13/data/NL/NL_",
        DISK_ROOT + "n4_nato_speech_LDC2006S13/data/DE/DE_",
    ]
    langs = ["UK","CA","NL","DE"]
    
    for idx,file in enumerate(inputs):
        all_short_segments = []
        
        wav_out_path = file + 'Audio_Speech_Segments'
        if os.path.exists(wav_out_path):
            os.system(f"rm -rf {wav_out_path}")
        os.mkdir(wav_out_path)
        
        for audio_file in glob(file+"Audio_Sphere/*.wav"):
            ts_file = inputs[idx]+'Trans/'+os.path.basename(audio_file).replace('wav','TRS')
            short_segments=extract_short_segments(ts_file, audio_file, wav_out_path, max_duration=30)   
            all_short_segments.extend(short_segments)    
        
        make_metadata(all_short_segments,langs[idx],out_file_path=f"metadata_{langs[idx]}.json")
        
        # speakers_split(all_short_segments) #! USED for counting and splitting speakers according to the gender and count for test and train set
        
        