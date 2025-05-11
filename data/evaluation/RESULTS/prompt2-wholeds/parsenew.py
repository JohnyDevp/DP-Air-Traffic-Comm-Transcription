import os
import re
import argparse

parser = argparse.ArgumentParser(description="Parse evaluation configuration")
parser.add_argument('--eval_dir', type=str, required=True, default=None)
parser.add_argument('--eval_files',  action="store_true")
args = parser.parse_args()

# Directory with evaluation files
eval_dir = args.eval_dir  # Change to your folder path
output_file = os.path.join(args.eval_dir,"summary_results.txt")
list_of_files = []
if args.eval_files:
    list_of_files = [
        '5B_eval_total_onlycalls_400.txt',
        '35B_eval_total_onlycalls_400.txt',
        '50CZB_eval_total_onlycalls_400.txt',
        'AG_eval_total_onlycalls_400.txt',
        'AG35B_eval_total_onlycalls_400.txt',
        'AG50CZB_eval_total_onlycalls_400.txt',
        'AG4B_eval_total_onlycalls_400.txt'
    ]
    # list_of_files = [
    #     '5B_eval_total_400.txt',
    #     '35B_eval_total_400.txt',
    #     '50CZB_eval_total_400.txt',
    #     'AG_eval_total_400.txt',
    #     'AG35B_eval_total_400.txt',
    #     'AG50CZB_eval_total_400.txt',
    #     'AG4B_eval_total_400.txt',
    # ]
    # list_of_files = [
    #     'eval_correct_5B_total.txt',
    #     'eval_correct_35B_total.txt',
    #     'eval_correct_50CZB_total.txt',
    #     'eval_correct_AG_total.txt',
    #     'eval_correct_AG_35B_total.txt',
    #     'eval_correct_AG_50CZB_total.txt',
    #     'eval_correct_AG_4B_total.txt',
    # ]
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

if True:
    with open(output_file, "w") as out_f:
        for file in list_of_files:
            with open(file, "r") as f:
                lines = f.readlines()               
                for line in lines:
                    if line.startswith("DATASET: allds |"):
                        _wer = re.search(r'WER: ([\d.]+)', line).group(1)
                        _loss = re.search(r'LOSS: ([\d.]+)', line).group(1)
                        _cal_wer = re.search(r'CALLSIGN WER: ([\d.]+)', line).group(1)
                        _call_count = re.search(r'CALLSIGN COUNT: (\d+)', line).group(1)    
                        _call_correct = re.search(r'CALLSIGN COMPLETELY CORRECT: (\d+)', line).group(1)
                        # Compute correctness ratio
                        # print(_call_count, _call_correct)
                        correct_ratio = 100.0 * float(_call_correct) / float(_call_count)
                        
                        out_f.write(f"===== {''.join(os.path.basename(file).split('_')[0])} =====\n")
                        out_f.write(f"Weighted WER: {float(_wer):.2f}%\n")
                        out_f.write(f"Weighted CALLSIGN WER: {float(_cal_wer):.2f}%\n")
                        out_f.write(f"CALLSIGNS COMPLETELY CORRECT: {_call_correct} ({correct_ratio:.2f}%)\n\n")
                        break

                # Output formatted lines
                # out_f.write(f"===== {file.split('/')[-2]} =====\n")
                # out_f.write(f"Weighted WER: {avg_wer:.2f}%\n")
                # out_f.write(f"Weighted CALLSIGN WER: {avg_callsign_wer:.2f}%\n")
                # out_f.write(f"CALLSIGNS COMPLETELY CORRECT: {total_callsigns}\n\n")
