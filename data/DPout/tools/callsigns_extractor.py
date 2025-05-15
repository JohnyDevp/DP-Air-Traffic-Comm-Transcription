import requests
from bs4 import BeautifulSoup
import json
from tqdm import tqdm

# Wikipedia page URL
url = "https://en.wikipedia.org/wiki/List_of_airline_codes"

# Fetch the page
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# Find all tables with class 'wikitable sortable'
tables = soup.find_all("table", class_="wikitable sortable")

# Extract ICAO codes and call signs
data = {}

for table in tables:
    rows = table.find_all("tr")[1:]  # Skip header row
    for row in tqdm(rows):
        cols = row.find_all("td")
        if len(cols) >= 3:  # Ensure there are enough columns
            icao_code = cols[1].text.strip()
            if (icao_code == 'n/a'): continue
            airline = cols[2].text.strip()
            call_sign = cols[3].text.strip()
            # data[call_sign] = {"icao": icao_code, "airline": airline} # STORE IN FORM CALLSIGN: {ICAO, AIRLINE}
            data[airline]   = {"icao": icao_code, "call_sign": call_sign}
            # STORE IN FORM ICAO: {CALLSIGN, AIRLINE}
            # if icao_code:  # Only keep rows where ICAO code is set
            #     if icao_code not in data:
            #         data[icao_code] = {"call_sign": call_sign, "airline": airline}
            #     else:
            #         if call_sign in data[icao_code]["call_sign"]:
            #             continue
            #         data[icao_code]["call_sign"] += ", " + call_sign
                    # data[icao_code]["airline"] += ", " + airline

# Save to JSON
with open("airline_icao.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

# Print some output
# print(json.dumps(data, indent=4, ensure_ascii=False))  # Show first 10 entries
