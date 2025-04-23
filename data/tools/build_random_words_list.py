# https://clanky.rvp.cz/clanek/o/z/1657/FILM-A-VYUKA.html
# https://www.branadovesmiru.eu/odborne-clanky/vykonove-lasery-cesta-ke-hvezdam-tezbe-rud-na-asteroidech-i-obrane-planety.html

import json
import random
import re
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def extract_words(text):
    # Split text into words, assuming words are separated by whitespace
    words = re.sub(r'\s+',' ', text).split(' ')
    for idx,word in enumerate(words):
        words[idx] = re.sub(r'[^\w\s]','',word).lower()
        if (words[idx] == "" or words[idx].isnumeric()): words.pop(idx)
        
    return list(set(words))

def save_to_json(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def main():
    input_file = 'czech_text.txt'  # Replace with your input file path
    output_file = 'random_czech_words.json'  # Replace with your output file path

    # Read the text file
    text = read_text_file(input_file)

    # Extract words from the text
    words = extract_words(text)

    # Shuffle the words randomly
    random.shuffle(words)

    # Save the shuffled words to a JSON file
    save_to_json(words, output_file)

if __name__ == '__main__':
    main()