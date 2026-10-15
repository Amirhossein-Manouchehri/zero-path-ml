"""Render the completed, fixed-test evaluation without retraining or selecting models."""
import json
import os
from src.dataset import PROJECT
os.environ.setdefault('MPLCONFIGDIR',str(PROJECT/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    folder=PROJECT/'models/L16_cnn_v1'
    report=json.loads((folder/'metrics.json').read_text())
    history=json.loads((folder/'history.json').read_text())
    rows=report['per_p']; p=np.array([r['p'] for r in rows]); solver=np.array([r['solver'] for r in rows])
    ci=np.array([r['solver_wilson_95'] for r in rows])
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    ax=axes[0,0]
    ax.errorbar(p,solver,yerr=[solver-ci[:,0],ci[:,1]-solver],fmt='o',color='black',capsize=2,label='Solver: 75 test realizations/p',markersize=3)
    colors=dict(density='#238b45',logistic='#8856a7',cnn='#2171b5')
    labels=dict(density='Negative-bond fraction',logistic='Individual-bond logistic',cnn='Lattice CNN (calibrated)')
    for name in colors:
        ax.plot(p,[r[name]['mean'] for r in rows],color=colors[name],label=labels[name])
    ax.set(xlabel='p = probability of a negative bond',ylabel='Mean percolation probability',ylim=(-.03,1.03),title='L=16: independent test ensemble')
    ax.legend(fontsize=8)
    ax=axes[0,1]
    ax.axhline(0,color='black',lw=1)
    for name in ['density','cnn']:
        difference=np.array([r[name]['difference'] for r in rows])
        bounds=np.array([r[name]['paired_bootstrap_difference_95'] for r in rows])
        ax.plot(p,difference,color=colors[name],label=labels[name])
        ax.fill_between(p,bounds[:,0],bounds[:,1],color=colors[name],alpha=.15)
    ax.set(xlabel='p',ylabel='Mean prediction − solver fraction',title='Paired bootstrap 95% intervals; fixed model')
    ax.legend(fontsize=8)
    ax=axes[1,0]
    with np.load(folder/'test_predictions.npz') as data:
        y=data['y']
        ax.plot([0,1],[0,1],color='gray',linestyle='--')
        for name in colors:
            q=data[name]; xs=[]; ys=[]
            for i in range(10):
                mask=(q>=i/10)&(q<((i+1)/10) if i<9 else q<=1)
                if mask.sum()>=10:
                    xs.append(q[mask].mean()); ys.append(y[mask].mean())
            ax.plot(xs,ys,'o-',color=colors[name],label=labels[name],markersize=4)
    ax.set(xlabel='Mean predicted probability in bin',ylabel='Observed positive fraction',xlim=(0,1),ylim=(0,1),title='Test calibration (bins with ≥10 realizations)')
    ax.legend(fontsize=8)
    ax=axes[1,1]
    ax.plot([r['epoch'] for r in history],[r['train_loss'] for r in history],label='Training')
    ax.plot([r['epoch'] for r in history],[r['validation_loss'] for r in history],label='Validation')
    ax.axvline(report['best_epoch'],color='gray',linestyle='--',label='Selected epoch')
    ax.set(xlabel='Epoch',ylabel='Binary cross-entropy',title='Checkpoint chosen using validation only')
    ax.legend(fontsize=8)
    for ax in axes.flat: ax.grid(alpha=.2)
    output=PROJECT/'figures/L16_cnn_v1_evaluation.png'
    fig.savefig(output,dpi=170);plt.close(fig)
    print(output,flush=True)

if __name__=='__main__': main()
