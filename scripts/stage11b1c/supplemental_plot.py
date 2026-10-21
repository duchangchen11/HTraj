"""CPU-only supplementary axis view: preserves every actual training observation."""
from scripts.stage11b1c.common import *
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    assert not torch.cuda.is_initialized()
    rows=list(csv.DictReader((OUT/'epochs.csv').open()))
    decision=json.loads((OUT/'final_decision.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    for k,label,color in [('train_loss','Total','#324E69'),('goal_loss','Goal','#B36C47'),('path_loss','Path','#538C7A')]:
        axes[0].plot([int(r['epoch']) for r in rows],[float(r[k]) for r in rows],'-o',ms=3,label=label,color=color)
        axes[1].plot([int(r['epoch']) for r in rows[1:]],[float(r[k]) for r in rows[1:]],'-o',ms=3,label=label,color=color)
    axes[0].set(yscale='log',title='All30 epochs (logarithmic loss axis)',ylabel='Original BCE x1000, target-weighted mean')
    axes[1].set(title='Epochs2-30 (linear loss axis)',ylabel='Original BCE x1000, target-weighted mean')
    for ax in axes:
        ax.set_xlabel('Actual completed epoch; no smoothing')
        ax.axvline(decision['best_epoch'],color='.5',linestyle=':',label='Validation ADE-selected epoch')
        ax.legend(fontsize=7)
    fig.text(.01,.01,'Y-Net a345ada | HTraj-YNet-SDD-Development-v1 | seed17 | K20 validation | loss is dimensionless\nBest checkpoint SHA: '+decision['best_checkpoint']['sha256'],fontsize=7)
    fig.tight_layout(rect=(0,.11,1,1))
    for ext in ['png','svg']:fig.savefig(FIG/f'07_training_loss_scale_view.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    assert not torch.cuda.is_initialized()
    print('Supplementary actual-loss axis views saved; no data smoothing or GPU use')

if __name__=='__main__':main()
