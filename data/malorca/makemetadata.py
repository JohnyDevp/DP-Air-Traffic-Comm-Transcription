# {
#   "filepath":str,
#   "full_ts":str,
#   "short_ts":str,
#   "language":str     
# }

import glob as glob
import os
from bs4 import BeautifulSoup
import re

def get_fullts(fullpath):
    with open(fullpath, "r") as f:
        return f.read().strip()

# Replace specific tags and their content
def replace_tag_with_word(text, tag, replacement):
    pattern = fr"<{tag}.*?>.*?</{tag}>"
    return re.sub(pattern, replacement, text)

def get_shortts_from_tra(fullpath_tra : str):
    with open(fullpath_tra, "r") as f:
        all_text = f.read()
        soup = BeautifulSoup(all_text, "html.parser")
        
        # check if cmd exists - if so, obtain the callsign from it
        cmd_path = fullpath_tra.replace(".tra", ".cmd")
        if os.path.exists(cmd_path):    
            with open(cmd_path, "r") as f:
                callsign = f.readline().split(' ')[0]
        else:
            callsign = ""
        
        if (callsign != ""):
            all_text = replace_tag_with_word(all_text, "callsign", callsign)
        # go through the text and replace some parts with numbers and callsign
        soup.find_all('w')

def makemetadata():
    PATH_TO_LISTINGS_OF_FILES = "/run/media/johnny/31c5407a-2da6-4ef8-95ec-d294c1afec38/MALORCA/DATA_ATC/VIENNA/DATA/dev12/wav.scp"
    DISK_PATH="/run/media/johnny/31c5407a-2da6-4ef8-95ec-d294c1afec38"
    out_data = []
    with open(PATH_TO_LISTINGS_OF_FILES, "r") as f:
        for line in f.readlines():
            # append path for the wavfile
            #! path prefix leading to root of my disk, where the data are stored, to LOWWXX, from where the path is unique for the file
            #! this is done because the path in the wav.scp is wrong
            path_prefix = "MALORCA/DATA_ATC/VIENNA/WAV_FILES"
            file_path=os.path.join(path_prefix, line.split('/VIENNA/')[1].strip())
            full_path_current_disk = os.path.join(DISK_PATH, file_path)
            
            out_data.append({"file":file_path})
            
            # append the full transcription
            if (os.path.exists(full_path_current_disk.replace(".wav", ".cmd"))):
                print(full_path_current_disk)
            # 1. check whether .tra - if it does, check for the .cmd file with callsign
            tra_path = full_path_current_disk.replace(".wav", ".tra")
            # if os.path.exists(tra_path):
            #     print(tra_path)

                # break

if __name__ == "__main__":
    makemetadata()
    pass