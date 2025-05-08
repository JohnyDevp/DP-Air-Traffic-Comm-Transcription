import re

# =================================================================================================================
# INSERT YOUR CHECKPOINT DATA HERE
# =================================================================================================================
text = """
#### EVAL MODEL openai/whisper-medium ####
******** Evaluation results ********
DATASET: atco_en_ruzyne | WER: 73.43565525383707 LOSS: 1.4866019146783012 CALLSIGN WER: 88.07443365695794 CALLSIGN COUNT: 103 CALLSIGN COMPLETELY CORRECT: 0
DATASET: atco_en_stefanik | WER: 71.48732008224812 LOSS: 1.3571279346942902 CALLSIGN WER: 87.82674772036472 CALLSIGN COUNT: 94 CALLSIGN COMPLETELY CORRECT: 1
DATASET: atco_en_zurich | WER: 81.8025182239894 LOSS: 1.6739440872555686 CALLSIGN WER: 88.19016770430194 CALLSIGN COUNT: 566 CALLSIGN COMPLETELY CORRECT: 11
"""
# =================================================================================================================

lengths = {
    'atco_en_ruzyne': 718,
    'atco_en_stefanik': 629,
    'atco_en_zurich': 2996,
}

extracted_data = {}
for line in text.splitlines():
    match = re.search(r"DATASET:\s*(\S+)\s*\|\s*WER:\s*([0-9.]+).*?CALLSIGN WER:\s*([0-9.]+).*?CALLSIGN COUNT:\s*([0-9.]+).*?CALLSIGN COMPLETELY CORRECT:\s*([0-9.]+)", line)
    if match:
        extracted_data[match.group(1)] = {
            "wer": float(match.group(2)),
            "callsign_wer": float(match.group(3)),
            "callsign_count": int(match.group(4)),
            "callsign_completely_correct": int(match.group(5)),
        }
        
# compute total everyhing
total_wer = 0
total_callsign_wer = 0
total_callsign_count = 0
total_callsign_completely_correct = 0
for key, value in extracted_data.items():
    if key in lengths:
        total_wer += value["wer"] * (lengths[key] / sum(lengths.values()))
        total_callsign_wer += value["callsign_wer"] * (lengths[key] / sum(lengths.values()))
        total_callsign_count += value["callsign_count"]
        total_callsign_completely_correct += value["callsign_completely_correct"]
        
# Print the results
print(f"FILE:{text.splitlines()[1]}")
print(f"Total WER: \t\t\t\t{total_wer:.2f}")
print(f"Total CALLSIGN WER: \t\t\t{total_callsign_wer:.2f}")
print("Total CALLSIGN COUNT: \t\t\t", total_callsign_count)
print(f"Total CALLSIGN COMPLETELY CORRECT: \t {total_callsign_completely_correct} ({total_callsign_completely_correct / total_callsign_count:.2%})")
# print(extracted_data)