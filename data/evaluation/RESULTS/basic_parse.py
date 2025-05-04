import os, json, re
import numpy as np
import matplotlib.pyplot as plt
import time
from argparse import ArgumentParser

# number of examples in dataset
# num_examples = {
#     'atco_en_ruzyne': 70,
#     'atco_en_stefanik': 53,
#     'atco_en_zurich': 412,
# }
lengths = {
    'atco_en_ruzyne': 718,
    'atco_en_stefanik': 629,
    'atco_en_zurich': 2996,
}

def compute_wer_loss(data):
    out = {}
    name=None
    for line in data:
        if ("EVAL MODEL" in line):
            iter_number = re.search(r'(\d)\_iter', line)
            iter_number = iter_number.group(1) if iter_number is not None else '1'
            name=line.strip().split(' ')[-2].split('/')[-1]
            name = iter_number + '_iter_' + name.replace('checkpoint-','') 
            # print(name)
            # print(line)
            out[name]={}
        elif name in out:
            dataset = re.search(r'DATASET: ([a-zA-Z_]+)', line)
            wer = re.search(r'WER: ([\d.]+)', line)
            loss = re.search(r'LOSS: ([\d.]+)', line)
            cal_wer = re.search(r'CALLSIGN WER: ([\d.]+)', line)
            if dataset is None or wer is None and loss is None:
                continue
            out[name][dataset.group(1)]={
                'wer': float(wer.group(1) if wer is not None else 0.0),
                'loss': float(loss.group(1) if loss is not None else 0.0),
            }    
            if cal_wer is not None:
                out[name][dataset.group(1)].update({'cal_wer': float(cal_wer.group(1))})
    return out

def compute_total_wer(out):
    # total wer
    checkpoints =[]
    wer = {'total':[]}
    cal_wer = {'total':[]}
    loss = {'total':[]}
    for checkpoint,ds_set in out.items():
        checkpoints.append(checkpoint)
        total_wer = 0
        total_loss = 0
        total_cal_wer = 0
        for ds_name,wer_loss in ds_set.items():
            if ds_name not in lengths.keys():
                continue
            if ds_name not in wer:
                wer[ds_name] = []
            if ds_name not in loss:
                loss[ds_name] = []
            if ds_name not in cal_wer:
                cal_wer[ds_name] = []
            wer[ds_name].append(wer_loss['wer'])
            loss[ds_name].append(wer_loss['loss'])
            if 'cal_wer' in wer_loss:
                cal_wer[ds_name].append(wer_loss['cal_wer'])
                total_cal_wer += wer_loss['cal_wer'] * lengths[ds_name]
            total_wer += wer_loss['wer'] * lengths[ds_name]
            total_loss += wer_loss['loss'] * lengths[ds_name]
        # average wer and loss
        wer['total'].append(total_wer / sum(lengths.values()))
        loss['total'].append(total_loss / sum(lengths.values()))
        if 'cal_wer' in wer_loss:
            cal_wer['total'].append(total_cal_wer / sum(lengths.values()))
    return checkpoints, wer, loss, cal_wer

def myplot(checkpoints, 
           wer, loss, cal_wer, ax, title='', src_file='',
           plot_total_wer=True, plot_partial_wer=True, 
           plot_total_loss=True, plot_partial_loss=False, 
           plot_total_cal_wer=False, plot_partial_cal_wer=False,
           display_left_axis_label=False, display_right_axis_label=False
    ):
    best_wer={}
    legend_lines = []
    x_arange = np.arange(start=1,stop=len(checkpoints)+1)
    line_width_total = 2
    line_width_partial = 1.5
    # plot WER
    for ds_name in wer:
        best_wer[ds_name] = wer[ds_name][np.argmin(wer['total'])]  
        if ds_name == 'total':
            if plot_total_wer:
                line, = ax.plot(x_arange,wer['total'],linewidth=line_width_total, marker='',color='purple', label='Total WER', zorder=10)
                legend_lines.append([line, 'Total WER'])
            continue
        if plot_partial_wer:
            line, =ax.plot(x_arange,wer[ds_name],linestyle='--',linewidth=line_width_partial, marker='',alpha=0.7, label=ds_name)
            legend_lines.append([line, ds_name])
    
    ax.set_xticks([1,int(0.25*len(checkpoints)),int(0.5*len(checkpoints)),int(0.75*len(checkpoints)),len(checkpoints)])
    ax.set_xlim(0.5, len(checkpoints)+0.5)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    if display_left_axis_label:
        ax.set_ylabel('WER %', fontsize=17)
    ax.set_xlabel('Epoch', fontsize=17)
    ax.set_title('Word Error Rate' if title == '' else title,fontsize=19)
    
    # plot CAL WER
    best_calwer = {}
    for ds_name in cal_wer:
        best_calwer[ds_name] = cal_wer[ds_name][np.argmin(wer['total'])]  
        if ds_name == 'total':
            if plot_total_cal_wer:
                line, = ax.plot(x_arange,cal_wer['total'],linewidth=line_width_total, marker='',color='red', label='Total CAL WER', zorder=10)
                legend_lines.append([line, 'Total CAL WER'])
            continue
        if plot_partial_cal_wer:
            line, = ax.plot(x_arange,cal_wer[ds_name],linestyle='--', linewidth=line_width_partial,marker='',alpha=0.7, label=ds_name)
            legend_lines.append([line, ds_name])
            
    # plot loss
    ax2 = ax.twinx()
    for ds_name in loss:
        if ds_name == 'total':
            if plot_total_loss:
                line, = ax2.plot(x_arange,loss['total'],linewidth=line_width_total, label='Total Loss')
                legend_lines.append([line, 'Total Loss'])
            continue
        if plot_partial_loss:
            line, = ax2.plot(x_arange,loss[ds_name],linestyle='--',linewidth=line_width_partial,alpha=0.7, label=ds_name)
            legend_lines.append([line, ds_name])
    
    if display_right_axis_label:
        ax2.set_ylabel('Loss', fontsize=17)
        
    ax2.tick_params(axis='y', labelsize=16)
    
    with open(os.path.dirname(src_file) + '/' + os.path.basename(src_file).split('.')[0] + '.best', 'w') as f:
        f.write(f"Best WER - epoch {np.argmin(wer['total'])+1}, {checkpoints[np.argmin(wer['total'])]}:\n")
        for ds_name in best_wer:
            # print (ds_name, best[ds_name])
            f.write(f"{ds_name} {best_wer[ds_name]}\n")
        
        f.write(f"Best CALLSIGN WER - epoch {np.argmin(cal_wer['total'])+1}, {checkpoints[np.argmin(cal_wer['total'])]}:\n")
        for ds_name in best_calwer:
            # print (ds_name, best[ds_name])
            f.write(f"{ds_name} {best_calwer[ds_name]}\n")
            
    return legend_lines

if __name__ == '__main__':

    argparse = ArgumentParser(description='Parse WER and Loss from log files')
    argparse.add_argument('--file', type=str, default='.', help='Directory containing the log files')
    args = argparse.parse_args()
    
    TITLES = [
        'Word Error Rate',
        'Callsign Word Error Rate',
        'Loss'
    ]   
    TO_PLOT = [
        'wer',
        'cal_wer',
        'loss'
    ] 
    fig, axs = plt.subplots(3, 1, figsize=(15, 15))

    data = open(args.file).readlines()
    out=compute_wer_loss(data)
    checkpoints, wer, loss, cal_wer = compute_total_wer(out)
    
    for idx in range(3):
        legend_lines = myplot(
            checkpoints, wer, loss, cal_wer, axs[idx],title=TITLES[idx], src_file=args.file,
            plot_total_wer=TO_PLOT[idx]=='wer', plot_partial_wer=TO_PLOT[idx]=='wer',
            plot_total_loss=TO_PLOT[idx]=='loss', plot_partial_loss=TO_PLOT[idx]=='loss',
            plot_total_cal_wer=TO_PLOT[idx]=='cal_wer', plot_partial_cal_wer=TO_PLOT[idx]=='cal_wer',
            display_left_axis_label=True, #if idx == 0 else False,
            display_right_axis_label=True #if idx == NUM_OF_PLOTS-1 else False,
        )
        
    legend_dict = {
        'atco_en_ruzyne': 'Ruzyne',
        'atco_en_stefanik': 'Stefanik',
        'atco_en_zurich': 'Zurich',
    }

    fig.legend([leg[0] for leg in legend_lines], [legend_dict[leg[1]] if leg[1] in legend_dict else leg[1] for leg in legend_lines], loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.00),fontsize=19)
    
    fig.tight_layout()
    
    # build output directory
    OUT_FILE = str(args.file).replace('.txt','')
    to_add = ''
    if (os.path.exists(OUT_FILE+'.png')):
        to_add = str(time.time())
    fig.savefig(f'{OUT_FILE}{'_'+to_add if to_add else ''}.png', dpi=300, bbox_inches='tight')