import os
import matplotlib.pyplot as plt
import argparse
import time, re

argparser = argparse.ArgumentParser(description="Plot WER and CALLSIGN WER")
argparser.add_argument(
    "--dir",
    type=str,
    default="./",
    help="Directory containing folders like '5B', '35B', etc.",
)
argparser.add_argument('--eval_files',action='store_true', help='Use eval files instead of best files')
argparser.add_argument('--name',type=str,default='wer',help='Name of the model plotted')
args = argparser.parse_args()
# Directory containing folders like "5B", "35B", etc.
root_dir = args.dir  # change if needed

# Weights for computing weighted average
# lengths = {
#     'atco_en_ruzyne': 718,
#     'atco_en_stefanik': 629,
#     'atco_en_zurich': 2996,
# }

# Normalize weights
# total_len = sum(lengths.values())
# weights = {k: v / total_len for k, v in lengths.items()}

# To store results
wer_data = {}
callsign_data = {}

files = []

if args.eval_files:
    files = [
        'AG/eval.best',
        'AG4B/eval.best',
        'AG35B/eval.best',
        'AG50CZB/eval.best',
        '5B/eval.best',
        '35B/eval.best',
        '50CZB/eval.best',
    ]
    files = [os.path.join(root_dir, f) for f in files]
else:
    for dirname in os.listdir(root_dir):
        if ('-' in dirname) or ('_' in dirname):
            continue
        file_to_append = os.path.join(root_dir, dirname, "eval.best")
        if not os.path.exists(file_to_append):
            file_to_append = os.path.join(root_dir, dirname, "eval_wholeds.best")
        files.append(file_to_append)

print(files)
   
# Traverse directories
for filepath in files:
    dirpath, filename = os.path.split(filepath)
    # dirname = filename.split("_")[0] if args.eval_files else os.path.basename(dirpath)
    dirname = os.path.basename(dirpath)
    # dirpath = os.path.join(root_dir, dirname)
    # filepath = os.path.join(dirpath, "eval.best")
    
    if os.path.isdir(dirpath) and os.path.exists(filepath):
        with open(filepath, "r") as f:
            lines = f.readlines()

        # Extract WER values
        wer_values = {}
        callsign_values = {}

        if False:
            wer_data[dirname] = {}
            callsign_data[dirname] = {}
            for line in lines[8:11]:
                match = re.search(r"DATASET:\s*(\S+)\s*\|\s*WER:\s*([0-9.]+).*?CALLSIGN WER:\s*([0-9.]+)", line)
                if match:
                    dataset = match.group(1)
                    wer = float(match.group(2))
                    callsign_wer = float(match.group(3))
                    wer_data[dirname][dataset] = wer
                    callsign_data[dirname][dataset] = callsign_wer
            wer_data[dirname]["weighted"] = sum(wer_data[dirname][k] * weights[k] for k in weights)
            callsign_data[dirname]["weighted"] = sum(callsign_data[dirname][k] * weights[k] for k in weights)
            continue
        
        else:
            # if "Best WER" in lines[0]:
            #     for line in lines[1:5]:
            #         name, val = line.split()
            #         wer_values[name] = float(val)

            # if "Best CALLSIGN WER" in lines[5]:
            #     for line in lines[6:]:
            #         name, val = line.split()
            #         callsign_values[name] = float(val)

            # Compute weighted average
            
            # wer_values["weighted"] = sum(wer_values[k] * weights[k] for k in weights)
            # callsign_values["weighted"] = sum(callsign_values[k] * weights[k] for k in weights)
            wer_data[dirname] = {
                'total': lines[1].split()[1],
                'atco_en_ruzyne': lines[2].split()[1],
                'atco_en_stefanik': lines[3].split()[1],
                'atco_en_zurich': lines[4].split()[1],
            }
            callsign_data[dirname] = {
                'total': lines[6].split()[1],
                'atco_en_ruzyne': lines[7].split()[1],
                'atco_en_stefanik': lines[8].split()[1],
                'atco_en_zurich': lines[9].split()[1],
            }
            # wer_data[dirname] = wer_values
            # callsign_data[dirname] = callsign_values

# Sort keys consistently
systems = wer_data.keys()
labels = ['atco_en_ruzyne', 'atco_en_stefanik', 'atco_en_zurich', 'total']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Plotting function
def plot_metric(data, title, name='wer'):
    x = range(len(data))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    legend_lines = []
    for i, label in enumerate(labels):
        vals = [float(data[sys][label]) for sys in systems]
        line =ax.bar([p + i * width for p in x], vals, width=width, label=label, color=colors[i])
        legend_lines.append((line, label))
        if label == 'total':
            vals_zurich = [float(data[sys]['atco_en_zurich']) for sys in systems]
            for j, val in enumerate(vals):
                x_pos = j + i * width
                y_top = val
                line_height = abs(val-vals_zurich[j]) + 0.7  # adjust height of the line in data units (e.g., WER percentage points)

                # Draw vertical line ("stem")
                ax.plot([x_pos, x_pos], [y_top, y_top + line_height], color='black', linewidth=1)

                # Draw text above the line
                ax.text(x_pos, y_top + line_height + 0.3, f"{val:.2f}%", ha='center', va='bottom',
                        fontsize=15, rotation=0)

    ax.set_xticks([p + 1.5 * width for p in x])
    ax.set_ylim(top=max(max(float(data[sys]['total']) for sys in systems) + 5, ax.get_ylim()[1]))
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    
    ax.set_xticklabels(systems)
    ax.set_ylabel("WER (%)", fontsize=17)
    ax.set_xlabel("Typ promptu", fontsize=17)
    ax.set_title(title, fontsize=19)

    
    legend_dict = {
        'atco_en_ruzyne': 'Ruzyně',
        'atco_en_stefanik': 'Štefánik',
        'atco_en_zurich': 'Zurich',
        'total': 'Total WER',
    }

    ax.legend([leg[0] for leg in legend_lines], [legend_dict[leg[1]] if leg[1] in legend_dict else leg[1] for leg in legend_lines], loc='upper center', ncol=4, bbox_to_anchor=(0.5, -0.14),fontsize=19)
    
    fig.tight_layout()
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # build output directory
    OUT_FILE = str(args.dir)
    to_add = ''
    path = os.path.join(OUT_FILE, name)
    if (os.path.exists(path+'.png')):
        to_add = str(time.time())
    fig.savefig(f'{path}{'_'+to_add if to_add else ''}.png', dpi=300, bbox_inches='tight')
    
    # plt.show()

# Plot WER and CALLSIGN WER
title_add = 'P-'+os.path.basename(args.dir).split('-')[0].upper()
if args.eval_files:
    title_add = os.path.basename(args.dir).split('-')[1].upper() #+ ' - BASE'
if args.name:
    title_add = args.name.upper()
plot_metric(wer_data, f"WER PŘEPISŮ, {title_add}", name='wer_ts')
plot_metric(callsign_data, f"WER VOLACÍCH ZNAKŮ, {title_add}", name='wer_cal')
