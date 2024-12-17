import re

# Initialize an empty list to store the results
results = []

# Path to your file
file = """
<Speaker id="spk17" name="UK_01_Z8U" check="no" dialect="native" accent="" scope="local" type="female"/>
<Speaker id="spk18" name="UK_06_T1L" check="no" dialect="native" accent="" scope="local" type="male"/>
<Speaker id="spk19" name="UK_11_9HD" check="no" dialect="native" accent="" scope="local" />
<Speaker id="spk20" name="UK_01_F3J" check="no" dialect="native" accent="" scope="local" type="female"/>
"""

# # Regex pattern to match id, optional type, and name in <Speaker> tags
# # pattern = r'<Speaker\s+[^>]*id="(.*?)".*?name="(.*?)".*?(?:type="(.*?)")?'
# # pattern = r'<Speaker\s+[^>]*id="(.*?)".*?name="(.*?)".*?(?:type="(.*?)")'
# pattern = r'<Speaker\s+[^>]*id="(.*?)"[^>]*name="(.*?)"[^>]*(?:type="(.*?)")?[^>]*>'
# # Open and read the file line by line
# for line in file.split("\n"):
#     # Search for <Speaker> tags with id, optional type, and name
#     match = re.search(pattern, line)
#     if match:
#         speaker_id = match.group(3)  # Extracted id
#         # speaker_type = match.group(2) if match.group(2) else "N/A"  # Extracted type (optional)
#         # speaker_name = match.group(3)  # Extracted name
#         print(speaker_id)
#         # Append to results as a dictionary
#         # results.append({"id": speaker_id, "type": speaker_type, "name": speaker_name})

# # Print the results
# for speaker in results:
#     print(speaker)

import re

# Initialize an empty list to store the results
results = []

# Path to your file
file_path = 'large_file.xml'

# Updated regex pattern with type at the end
pattern = r'<Speaker\s+[^>]*id="(.*?)"\s+[^>]*name="(.*?)"(?:\s+[^>]*type="(.*?)")?'

# Open and read the file line by line
with open(file_path, 'r') as file:
    for line in file:
        # Search for <Speaker> tags
        match = re.search(pattern, line)
        if match:
            speaker_id = match.group(1)  # Extracted id
            speaker_name = match.group(2)  # Extracted name
            speaker_type = match.group(3) if match.group(3) else "N/A"  # Optional type
            
            # Append to results as a dictionary
            results.append({"id": speaker_id, "name": speaker_name, "type": speaker_type})

# Print the results
for speaker in results:
    print(speaker)

