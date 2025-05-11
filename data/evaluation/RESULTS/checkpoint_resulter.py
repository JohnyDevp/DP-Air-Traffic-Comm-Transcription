import re

# =================================================================================================================
# INSERT YOUR CHECKPOINT DATA HERE
# =================================================================================================================
text = """
#### EVAL MODEL /mnt/scratch/tmp/xholan11/models/allds-atcoen-full/mypar/checkpoint-400 ####
******** Evaluation results ********
DATASET: atco_en_ruzyne | WER: 17.81354051054384 LOSS: 0.7172485888004303 CALLSIGN WER: 11.233766233766234 CALLSIGN COUNT: 77 CALLSIGN COMPLETELY CORRECT: 46
DATASET: atco_en_stefanik | WER: 17.463352453792226 LOSS: 0.6538456777731577 CALLSIGN WER: 12.252747252747252 CALLSIGN COUNT: 78 CALLSIGN COMPLETELY CORRECT: 55
DATASET: atco_en_zurich | WER: 21.379310344827587 LOSS: 0.9145827541748682 CALLSIGN WER: 16.34990938632671 CALLSIGN COUNT: 508 CALLSIGN COMPLETELY CORRECT: 292
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