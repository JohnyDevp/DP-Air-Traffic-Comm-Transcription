import json, os, re, random

SAVE_PATH='./'
files = [
    'metadata_en_ruzyne_test.json',
    'metadata_en_stefanik_test.json',
    'metadata_en_zurich_test.json',
    'metadata_en_train.json',
    'metadata_fr_test.json',
    'metadata_fr_train.json',
    'metadata_other_lang_test.json'
]

to_shuffle = [
    "prompt_fullts_AG_4B",
    "prompt_fullts_AG_40B",
    "prompt_fullts_AG_50CZB",
    "prompt_shortts_AG_4B",
    "prompt_shortts_AG_40B",
    "prompt_shortts_AG_50CZB"
]

if __name__ == "__main__":
    for file in files:
        # Load the JSON file
        with open(file, "r") as f:
            data = json.load(f)
            
            # shuffle the prompts per each record
            for item in data:
                for key in to_shuffle:
                    if key in item:
                        # split string to array of elements according to the comma
                        prepared_arr = item[key].split(",")
                        prepared_arr = [re.sub(r'\s+',' ',i).strip().replace(',','') for i in prepared_arr]
                        if (len(prepared_arr) > 0):
                            random.shuffle(prepared_arr)
                            item[key] = ','.join(prepared_arr)
    
            # Save the modified data to a new JSON file
            with open(os.path.join(SAVE_PATH, file), "w") as f:
                json.dump(data, f, indent=4,ensure_ascii=False)