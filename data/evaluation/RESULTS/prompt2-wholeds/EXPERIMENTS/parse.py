import os, re ,argparse
import numpy as np

argparser = argparse.ArgumentParser(description="Plot WER and CALLSIGN WER")
argparser.add_argument('--dir',type=str,required=True,default=None,help='Directory containing folders like "5B", "35B", etc.')

out_file = os.path.join(argparser.parse_args().dir,"summary_results.txt")

files = [
    'noprompt_5Bwholeds_total.txt',
    'noprompt_35Bwholeds_total.txt',
    'noprompt_50CZB_total.txt',
    # 'noprompt_AGwholeds_total.txt',
    'noprompt_AG35Bwholeds_shuffled_total.txt',
    'noprompt_AG50CZBwholeds_total.txt',
    'noprompt_AG4Bwholeds_total.txt',
    # 'AGvsBwholeds_total.txt',
    # 'BvsAGwholeds_total.txt',
]

for file in files:
    files[files.index(file)] = os.path.join(argparser.parse_args().dir, file)

out_f = open(out_file, 'w')
for file in files:
    with open(file, 'r') as f:
        # DATASET: allds | WER: 24.916270051119337 LOSS: 0.9769929029323436 CALLSIGN WER: 24.321028105431775 CALLSIGN COUNT: 763 CALLSIGN COMPLETELY CORRECT: 312
        alldata = re.search(r"DATASET: \w+ \| WER: ([\d.]+) LOSS: ([\d.]+) CALLSIGN WER: ([\d.]+) CALLSIGN COUNT: (\d+) CALLSIGN COMPLETELY CORRECT: (\d+)", f.read())
        _wer = alldata.group(1)
        _loss = alldata.group(2)
        _cal_wer = alldata.group(3)
        _call_count = alldata.group(4)
        _call_correct = alldata.group(5)
        
        out_f.write(f"===== {os.path.basename(file).replace('wholeds','')} =====\n")
        out_f.write(f"Weighted WER: {float(_wer):.2f}%\n")
        out_f.write(f"Weighted CALLSIGN WER: {float(_cal_wer):.2f}%\n")
        out_f.write(f"CALLSIGNS COMPLETELY CORRECT: {_call_correct} ({(float(_call_correct)/float(_call_count)):.2f}%)\n\n")

out_f.close()