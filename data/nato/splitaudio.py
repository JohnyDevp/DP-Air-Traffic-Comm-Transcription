import os
from librosa import ex
from pydub import AudioSegment 
import re
from bs4 import BeautifulSoup
from scipy.io import wavfile
import json
from rapidfuzz import process

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

def get_shortts(full_ts, what : str ="alphanum"):
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
    
    words : list[str] = full_ts.strip().split(' ')
    for word in words:
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

def extract_short_segments(input_file, output_file, max_duration=30):
    """
    Extracts segments from the input file where the duration between <Sync> tags is less than max_duration.
    
    :param input_file: Path to the input file.
    :param output_file: Path to save the filtered content.
    :param max_duration: Maximum duration in seconds between <Sync> tags to retain content.
    """
    with open(input_file, "r", encoding="utf-8") as file:
        content = file.read()
    
    # Regex to match <Sync time="..."/> and capture timestamps
    sync_pattern = r'<Sync time="([\d.]+)"/>'
    
    # Extract text between Sync tags
    split_content = re.split(sync_pattern, content)
    text_segments = split_content[1:-1]  # Skip the first and last split as it precedes and lasts the first <Sync>
    
    short_segments = []
    # Process pairs of Sync timestamps and their corresponding text
    for i in range(0, len(text_segments),2):
        if (i+2) >= len(text_segments):
            break
        current_time = float(text_segments[i])
        next_time = float(text_segments[i + 2])
        text = text_segments[i + 1]
        
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
            short_segments.append({
                "start": current_time,
                "end": next_time,
                "text": text
            })

    return short_segments[:2]

def split_audio(short_segments,input_file):
    if (not os.path.exists(input_file)):
        return
    
    audio = AudioSegment.from_file(input_file)
    bsnm = os.path.basename(input_file).replace(".wav", "")
    metadata = []
    for idx,segment in enumerate(short_segments):
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]
        # Load the audio file
        # Extract the segment
        segment = audio[start*1000:end*1000]
        # Save the segment
        newaudiofilename = f"audio/{bsnm}_{idx}.wav"
        segment.export(newaudiofilename, format="wav")
        
        short_ts = get_shortts(text)
        metadata.append({
            "audio":newaudiofilename,
            "full_ts":text,
            "short_ts":short_ts,
            "prompt": None
        })
    
    with open("metadata.json", "w") as file:
        json.dump(metadata, file, indent=4)
        
if __name__ == "__main__":
    # Example Usage
    input_file = "/run/media/johnny/31c5407a-2da6-4ef8-95ec-d294c1afec38/n4_nato_speech_LDC2006S13/data/UK/UK_Trans/UK_001.TRS"
    output_file = "test.txt"
    short_segments=extract_short_segments(input_file, output_file, max_duration=30)
    input_file = "/run/media/johnny/31c5407a-2da6-4ef8-95ec-d294c1afec38/n4_nato_speech_LDC2006S13/data/UK/UK_Audio_Sphere/UK_001.wav"
    split_audio(short_segments, input_file)
    