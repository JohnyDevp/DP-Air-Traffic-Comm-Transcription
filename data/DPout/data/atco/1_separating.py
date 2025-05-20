from glob import glob
from bs4 import BeautifulSoup
from tqdm import tqdm  
import shutil, os, sys

def is_english_lang(xml_data):
    # returns True if more than 50% of the segments are in English, otherwise False
    soup = BeautifulSoup(xml_data, "xml")
    english_segments = 0
    num_of_segments = len(soup.find_all("segment"))
    for segment in soup.find_all("segment"):
        tags = segment.find('tags')
        if (tags):
            tag_en = tags.find("non_english")
            if (tag_en): # the tag is <non_english></non_english>, meaning 0 is english
                english_segments += 1 - int(tag_en.get_text())
    
    if (num_of_segments == 0):
        return True    
    else: 
        return english_segments / num_of_segments > 0.5

if __name__ == "__main__":
    # SET ROOT DIR ==============================================================
    if len(sys.argv) > 1:
        DISK_ROOT = sys.argv[1]
    else:
        DISK_ROOT=""
    FOLDER_NAME = 'ATCO2-ASRdataset-v1_final'
    # ==============================================================
    
    ATCO_FOLDER_PATH = os.path.join(DISK_ROOT, FOLDER_NAME)
    # YOU MAY CHANGE NEXT LINES,  BUT IT WILL AFFECT ALL PROCESS IN OTHER SCRIPTS
    # DATA (english) folders ==============================================================
    DATA_EN_DIR = os.path.join(ATCO_FOLDER_PATH, 'DATA') # root folder with all english files
    DATA_EN_nonEN_DIR = os.path.join(ATCO_FOLDER_PATH, 'DATA-data-nonEN') # folder with all non-english files from DATA folder
    os.makedirs(os.path.join(ATCO_FOLDER_PATH, 'DATA-data-nonEN'), exist_ok=True) # create a new directory for non-english files in the DATA folder
    # DATA (non-english) folders ==============================================================
    DATA_NONEN_DIR = os.path.join(ATCO_FOLDER_PATH, 'DATA_nonEN') # root folder with all non-english files
    DATA_nonEN_datanonen_EN_DIR = os.path.join(ATCO_FOLDER_PATH, 'DATA_nonEN-datanonen-EN') # folder with all non-english files from DATA folder
    os.makedirs(os.path.join(ATCO_FOLDER_PATH, 'DATA_nonEN-datanonen-EN'), exist_ok=True) # create a new directory for english files in the DATA_nonEN folder
    
    # MOVING FILES all non english files from DATA_EN_DIR to DATA_EN_nonEN_DIR 
    # ==============================================================
    count_en_moved_to_nonen = 0
    for file in tqdm(glob(f'{DATA_EN_DIR}/*.xml')):
        if (not is_english_lang(open(file, mode='r').read())):
            count_en_moved_to_nonen += 1
            file = file.removesuffix('.xml') # remove the suffix .xml to match all files with the same name
            for src_path in glob(f'{file}.*'):
                # Move the file
                shutil.move(src_path, DATA_EN_nonEN_DIR)
    
    # MOVING FILES all english files from DATA_NONEN_DIR to DATA_nonEN_datanonen_EN_DIR
    # ==============================================================
    count_nonen_moved_to_en = 0
    for file in tqdm(glob(f'{DATA_NONEN_DIR}/*.xml')):
        if (is_english_lang(open(file, mode='r').read())):
            count_nonen_moved_to_en += 1
            file = file.removesuffix('.xml') # remove the suffix .xml to match all files with the same name
            for src_path in glob(f'{file}.*'):
                # Move the file
                shutil.move(src_path, DATA_nonEN_datanonen_EN_DIR)
                
    print(f'Moved {count_en_moved_to_nonen} files from DATA_EN_DIR to DATA_EN_nonEN_DIR')
    print(f'Moved {count_nonen_moved_to_en} files from DATA_NONEN_DIR to DATA_nonEN_datanonen_EN_DIR')