import os
from librosa import ex
from pydub import AudioSegment 
import re
from bs4 import BeautifulSoup

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
    
    filtered_text = []
    # Process pairs of Sync timestamps and their corresponding text
    for i in range(0, len(text_segments),2):
        if (i+2) >= len(text_segments):
            break
        current_time = float(text_segments[i])
        next_time = float(text_segments[i + 2])
        text = text_segments[i + 1]
    
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
            filtered_text.append({
                "start": current_time,
                "end": next_time,
                "text": text
            })

    print(filtered_text[0])

if __name__ == "__main__":
    # Example Usage
    input_file = "/run/media/johnny/31c5407a-2da6-4ef8-95ec-d294c1afec38/n4_nato_speech_LDC2006S13/data/UK/UK_Trans/UK_001.TRS"
    output_file = "test.txt"
    extract_short_segments(input_file, output_file, max_duration=30)
    # print(res.__len__())
    pass