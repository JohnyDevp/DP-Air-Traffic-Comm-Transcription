import os
import re
import argparse

# Dataset weights (number of examples per dataset)
lengths = {
    'atco_en_ruzyne': 718,
    'atco_en_stefanik': 629,
    'atco_en_zurich': 2996,
}

parser = argparse.ArgumentParser(description="Parse evaluation configuration")
parser.add_argument('--eval_dir', type=str, required=True, default=None)
parser.add_argument('--eval_files',  action="store_true")
args = parser.parse_args()

# Directory with evaluation files
eval_dir = args.eval_dir  # Change to your folder path
output_file = os.path.join(args.eval_dir,"summary_results.txt")

# Regex to extract info
pattern = re.compile(
    r"DATASET: (\w+) \| WER: ([\d.]+) LOSS: ([\d.]+) CALLSIGN WER: ([\d.]+) "
    r"CALLSIGN COUNT: (\d+) CALLSIGN COMPLETELY CORRECT: (\d+)"
)

# 35B_eval.txt  50CZB_eval.txt  5B_eval.txt  AG35B_eval.txt  AG4B_eval.txt  AG50CZB_eval.txt  AG_eval.txt
list_of_files = []
if args.eval_files:
    list_of_files = [
        # '5B_eval_wholeds.txt',
        # '35B_eval_wholeds.txt',
        # '50CZB_eval_wholeds.txt',
        'AG_eval_wholeds.txt',
        'AG35B_eval_wholeds.txt',
        'AG50CZB_eval_wholeds.txt',
        'AG4B_eval_wholeds.txt',
    ]
else:
    for dir in os.listdir(eval_dir):
        dir_path = os.path.join(eval_dir, dir)
        if os.path.isdir(dir_path):
            for file in os.listdir(dir_path):
                if file.__contains__("eval") and file.endswith(".best"):
                    list_of_files.append(os.path.join(dir_path, file))
        else:
            if dir.__contains__("eval") and dir.endswith(".best"):
                list_of_files.append(os.path.join(eval_dir, dir))


with open(output_file, "w") as out_f:
    for fname in list_of_files:
        if not fname.endswith(".txt"):
            continue

        with open(fname, "r") as f:
            text = f.read()

        matches = pattern.findall(text)
        if not matches:
            continue

        total_weight = 0
        weighted_wer = 0
        weighted_callsign_wer = 0
        total_callsigns = 0
        total_correct_callsigns = 0

        for dataset, wer, loss, cs_wer, cs_count, cs_correct in matches:
            wer = float(wer)
            cs_wer = float(cs_wer)
            cs_count = int(cs_count)
            cs_correct = int(cs_correct)

            weight = lengths.get(dataset, 0)
            total_weight += weight
            weighted_wer += wer * weight
            weighted_callsign_wer += cs_wer * weight
            total_callsigns += cs_count
            total_correct_callsigns += cs_correct

        avg_wer = weighted_wer / total_weight
        avg_callsign_wer = weighted_callsign_wer / total_weight
        correct_ratio = total_correct_callsigns / total_callsigns * 100

        out_f.write(f"===== {fname} =====\n")
        out_f.write(f"Weighted WER: {avg_wer:.2f}%\n")
        out_f.write(f"Weighted CALLSIGN WER: {avg_callsign_wer:.2f}%\n")
        out_f.write(f"CALLSIGNS COMPLETELY CORRECT: {total_correct_callsigns} / {total_callsigns} ({correct_ratio:.2f}%)\n\n")
