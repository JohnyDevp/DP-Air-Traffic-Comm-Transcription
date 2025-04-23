import re, json
from bs4 import BeautifulSoup, ResultSet

meta_files=[]
ROOT = '/'
for file in meta_files:
    js = json.load(open(ROOT+file))
    xml_files = [rec[''].replace('wav','xml') for path in js ]

for file in files:
    xml_content = open(file,'r').read()
    bs=BeautifulSoup(xml_content,features='xml')    
    for segment in bs.find_all('segment'):
        full_ts_with_tags = segment.find('text').text
        # now obtain all words in callsigns
        pattern= r'\[#callsign\](.*?)\[/#callsign\]'
        all_callsigns = re.findall(pattern, full_ts_with_tags)
        all_callsigns_length = 0
        for callsign in all_callsigns:
            # normalize callsign
            callsign = re.sub(r'\s+', ' ', callsign)
            all_callsigns_length += len(callsign)
        # remove all callsigns tags with its content
        full_ts_without_callsigns = re.sub(pattern, ' ', full_ts_with_tags)
        # now remove all other tags
        full_ts_clean = re.sub(r'\[[^\]]*\]',' ',full_ts_without_callsigns)
        # clean spaces
        full_ts_clean = re.sub(r'\s+',' ',full_ts_clean)
