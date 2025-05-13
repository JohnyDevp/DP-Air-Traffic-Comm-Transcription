import os, re ,argparse
import numpy as np

argparser = argparse.ArgumentParser(description="Plot WER and CALLSIGN WER")
argparser.add_argument('--dir',type=str,required=True,default=None,help='Directory containing folders like "5B", "35B", etc.')

out_file = os.path.join(argparser.parse_args().dir,"summary_results.txt")

files = [
    '5B/eval.txt',
    '5B/eval_total.txt',
    '35B/eval.txt',
    '35B/eval_total.txt',
    '50CZB/eval.txt',
    '50CZB/eval_total.txt',
    'AG/eval.txt',
    'AG/eval_total.txt',
    'AG4B/eval.txt',
    'AG4B/eval_total.txt',
    'AG35B/eval.txt',
    'AG35B/eval_total.txt',
    'AG50CZB/eval.txt',
    'AG50CZB/eval_total.txt',
]

for file in files:
    files[files.index(file)] = os.path.join(argparser.parse_args().dir, file)

out_f = open(out_file, 'w')
for file in files:
    print(file)
    if 'eval.txt' in file:
        f= open(file, 'r').readlines()
        out_f.write(f"===== {file.split('/')[-2]} =====\n")
        # DATASET: allds | WER: 24.916270051119337 LOSS: 0.9769929029323436 CALLSIGN WER: 24.321028105431775 CALLSIGN COUNT: 763 CALLSIGN COMPLETELY CORRECT: 312
        for line in f[8:11]:
            alldata = re.search(r"DATASET: (\w+) \| WER: ([\d.]+) LOSS: ([\d.]+) CALLSIGN WER: ([\d.]+) CALLSIGN COUNT: (\d+) CALLSIGN COMPLETELY CORRECT: (\d+)", line)
            _dsset = alldata.group(1)
            _wer = alldata.group(2)
            _loss = alldata.group(3)
            _cal_wer = alldata.group(4)
            _call_count = alldata.group(5)
            _call_correct = alldata.group(6)
            # print(_call_correct, _call_count)
            out_f.write(f"  | {_dsset}\n")
            out_f.write(f"      | WER: {float(_wer):.2f}%\n")
            out_f.write(f"      | CALLSIGN WER: {float(_cal_wer):.2f}%\n")
            out_f.write(f"      | CALLSIGNS COMPLETELY CORRECT: {_call_correct} ({(float(_call_correct)/float(_call_count)):.2f}%)\n")
    if 'eval_total.txt' in file:
        with open(file, 'r') as f:
            # DATASET: allds | WER: 24.916270051119337 LOSS: 0.9769929029323436 CALLSIGN WER: 24.321028105431775 CALLSIGN COUNT: 763 CALLSIGN COMPLETELY CORRECT: 312
            alldata = re.search(r"DATASET: \w+ \| WER: ([\d.]+) LOSS: ([\d.]+) CALLSIGN WER: ([\d.]+) CALLSIGN COUNT: (\d+) CALLSIGN COMPLETELY CORRECT: (\d+)", f.read())
            _wer = alldata.group(1)
            _loss = alldata.group(2)
            _cal_wer = alldata.group(3)
            _call_count = alldata.group(4)
            _call_correct = alldata.group(5)
            
            out_f.write(f"Total WER: {float(_wer):.2f}%\n")
            out_f.write(f"Total CALLSIGN WER: {float(_cal_wer):.2f}%\n")
            out_f.write(f"Total CALLSIGNS COMPLETELY CORRECT: {_call_correct} ({(float(_call_correct)/float(_call_count)*100):.2f}%)\n\n\n")

out_f.close()