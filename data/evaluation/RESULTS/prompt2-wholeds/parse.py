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
output_file = os.path.join(args.eval_dir,"summary_results2.txt")

# Regex to extract info
pattern = re.compile(
    r"DATASET: (\w+) \| WER: ([\d.]+) LOSS: ([\d.]+) CALLSIGN WER: ([\d.]+) "
    r"CALLSIGN COUNT: (\d+) CALLSIGN COMPLETELY CORRECT: (\d+)"
)

# 35B_eval.txt  50CZB_eval.txt  5B_eval.txt  AG35B_eval.txt  AG4B_eval.txt  AG50CZB_eval.txt  AG_eval.txt
list_of_files = []
if args.eval_files:
    list_of_files = [
        '5B/eval.best',
        '35B/eval.best',
        '50CZB/eval.best',
        'AG/eval.best',
        'AG35B/eval.best',
        'AG50CZB/eval.best',
        'AG4B/eval.best',
    ]
    # list_of_files = [
    #     'noprompt_5Bwholeds.txt',
    #     'noprompt_35Bwholeds.txt',
    #     'noprompt_50CZB.txt',
    #     'noprompt_AGwholeds.txt',
    #     'noprompt_AG35Bwholeds_shuffled.txt',
    #     'noprompt_AG50CZBwholeds.txt',
    #     'noprompt_AG4Bwholeds.txt',
    # ]
    for idx,file in enumerate(list_of_files):
        list_of_files[idx] = os.path.join(eval_dir, file)
else:
    list_of_files = [
        'noprompt_5Bwholeds.txt',
        'noprompt_35Bwholeds.txt',
        'noprompt_50CZB.txt',
        'noprompt_AGwholeds.txt',
        'noprompt_AG35Bwholeds_shuffled.txt',
        'noprompt_AG50CZBwholeds.txt',
        'noprompt_AG4Bwholeds.txt',
    ]
    for dir in os.listdir(eval_dir):
        dir_path = os.path.join(eval_dir, dir)
        # for file in list_of_files:
        #     list_of_files.append(os.path.join(dir_path, file))

        if os.path.isdir(dir_path):
            for file in os.listdir(dir_path):
                if file.__contains__("eval") and file.endswith(".best"):
                    list_of_files.append(os.path.join(dir_path, file))
        else:
            if dir.__contains__("eval") and dir.endswith(".best"):
                list_of_files.append(os.path.join(eval_dir, dir))

print(list_of_files)
print(output_file)
print('==========================')
if False:
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

if True:
    with open(output_file, "w") as out_f:
        for file in list_of_files:
            with open(file, "r") as f:
                lines = f.readlines()

                # Extract values
                avg_wer = None
                avg_callsign_wer = None
                # total_correct_callsigns = None
                total_callsigns = None

                for line in lines:
                    if line.startswith("total "):
                        value = float(line.strip().split()[1])
                        if avg_wer is None:
                            avg_wer = value
                        elif avg_callsign_wer is None:
                            avg_callsign_wer = value
                        elif total_callsigns is None:
                            total_callsigns = int(value)
                    # elif "COMPLETELY CORRECT CALLSIGN" in line:
                        # Next total line will be correct_callsigns
                        # continue
                    # elif total_callsigns is not None and total_correct_callsigns is None:
                    #     total_correct_callsigns = int(line.strip().split()[1])

                # Compute correctness ratio
                # correct_ratio = 100.0 * total_correct_callsigns / total_callsigns

                # Output formatted lines
                out_f.write(f"===== {file.split('/')[-2]} =====\n")
                out_f.write(f"WER Ruzyne {float(lines[2].split(' ')[1]):.2f}%\n")
                out_f.write(f"WER Stefanik {float(lines[3].split(' ')[1]):.2f}%\n")
                out_f.write(f"WER Zurich {float(lines[4].split(' ')[1]):.2f}%\n")
                out_f.write(f"CALLSIGN WER Ruzyne {float(lines[7].split(' ')[1]):.2f}%\n")
                out_f.write(f"CALLSIGN WER Stefanik {float(lines[8].split(' ')[1]):.2f}%\n")
                out_f.write(f"CALLSIGN WER Zurich {float(lines[9].split(' ')[1]):.2f}%\n")
                out_f.write(f"WER: {avg_wer:.2f}%\n")
                out_f.write(f"CALLSIGN WER: {avg_callsign_wer:.2f}%\n")
                out_f.write(f"CALLSIGNS COMPLETELY CORRECT: {total_callsigns}\n\n")
