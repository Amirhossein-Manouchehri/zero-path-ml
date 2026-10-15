"""Evaluate the frozen multi-size run, with L=36 as the final unseen-size test."""
import hashlib
import json
import math
import os
from pathlib import Path
import time
import joblib
import numpy as np
from scipy.special import expit
import torch
from src.dataset import PROJECT,DEFAULT_SOLVER,load_solver,unpack_bonds,total_percolates_with_length
from src.lattice_cnn import lattice_channels
from src.multisize_cnn import MultiSizeCNN
from src.predict_lattice import load_model
from src.train_lattice import metrics
from src.train_multisize import source_path

ROOT=PROJECT/'models/multisize_cnn_v1'
OUT=PROJECT/'models/multisize_eval_v1'

def probabilities(model,X,L,c,threads=2):
    torch.set_num_threads(threads);inputs=lattice_channels(X,L)
    with torch.no_grad():z=torch.cat([model(batch) for batch in inputs.split(128)]).numpy()
    return expit(z/c['temperature']+c.get('intercept',0)+c.get('log_size_slope',0)*math.log(L/16)),expit(z)

def summarize(y,p,probabilities_by_name,L):
    rows=[];bootstrap_means={name:[] for name in probabilities_by_name};rng=np.random.default_rng(311+L)
    for value in np.unique(p):
        mask=p==value;labels=y[mask];n=len(labels);fraction=float(labels.mean())
        z=1.959963984540054;denom=1+z*z/n;center=(fraction+z*z/(2*n))/denom
        half=z*np.sqrt(fraction*(1-fraction)/n+z*z/(4*n*n))/denom
        row=dict(p=float(value),n=n,solver=fraction,solver_wilson_95=[center-half,center+half])
        for name,q in probabilities_by_name.items():
            residual=q[mask]-labels
            boot=residual[rng.integers(n,size=(3000,n))].mean(1)
            bootstrap_means[name].append(boot)
            row[name]=dict(probability=float(q[mask].mean()),bias=float(residual.mean()),
                paired_bootstrap_bias_95=np.quantile(boot,[.025,.975]).tolist())
        rows.append(row)
    return dict(per_p=rows,metrics={name:metrics(y,q) for name,q in probabilities_by_name.items()},
        curve_mae={name:float(np.mean([abs(row[name]['bias']) for row in rows])) for name in probabilities_by_name},
        signed_bias={name:float(np.mean([row[name]['bias'] for row in rows])) for name in probabilities_by_name},
        signed_bias_bootstrap_95={name:np.quantile(np.mean(boot,axis=0),[.025,.975]).tolist() for name,boot in bootstrap_means.items()})

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    path=ROOT/'cnn.pt';frozen_hash=hashlib.sha256(path.read_bytes()).hexdigest()
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    model=MultiSizeCNN().eval();model.load_state_dict(checkpoint['state_dict'])
    old,old_checkpoint=load_model()
    baseline=joblib.load(ROOT/'density_size_baseline.joblib')
    reports=[]
    for L in [16,20,24,28,32,30,36]:
        dataset=source_path(L)
        with np.load(dataset,allow_pickle=False) as d:
            X,y,p=d['X'],d['y'],d['p_sample'];meta=json.loads(d['metadata'].item())
            lengths=d['path_length'] if 'path_length' in d.files else None
        assert len(y)==meta['samples']==meta['target_samples']
        assert hashlib.sha256(DEFAULT_SOLVER.read_bytes()).hexdigest()==meta['solver_sha256']
        if L in [30,36]:ids=np.arange(len(y))
        else:
            with np.load(ROOT/'split_indices.npz',allow_pickle=False) as split:ids=split[f'L{L}_test']
        X,y,p=X[ids],y[ids],p[ids]
        cnn,raw=probabilities(model,X,L,checkpoint['calibration'])
        old_q,_=probabilities(old,X,L,dict(temperature=old_checkpoint['temperature']))
        features=np.column_stack([(X==-1).mean(1),np.full(len(X),math.log(L/16))])
        z=baseline.decision_function(features);c=checkpoint['baseline_calibration']
        baseline_q=expit(z/c['temperature']+c['intercept']+c['log_size_slope']*math.log(L/16))
        probs=dict(cnn=cnn,cnn_raw=raw,density_size=baseline_q,density_size_raw=expit(z),old_l16_cnn=old_q)
        report=summarize(y,p,probs,L)
        role='Final unseen-size test' if L==36 else 'Size validation (used for selection/calibration)' if L==30 else 'Development-size held-out rows; earlier results inspected'
        report.update(L=L,n=len(y),role=role,checkpoint_sha256=frozen_hash,dataset=str(dataset),
            dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest())
        if L==36:
            assert len(y)==1200 and len(np.unique(X,axis=0))==1200
            assert lengths is not None and np.array_equal(lengths==0,y==0)
            probabilities(model,X[:8],L,checkpoint['calibration'],threads=1)
            timings=[]
            for _ in range(3):
                start=time.perf_counter();q,_=probabilities(model,X,L,checkpoint['calibration'],threads=1)
                timings.append(time.perf_counter()-start)
            np.testing.assert_allclose(q,cnn,atol=2e-6,rtol=2e-5)
            reference=load_solver(DEFAULT_SOLVER);restored=[unpack_bonds(row,L) for row in X[:48]]
            total_percolates_with_length(restored[0],reference)
            start=time.perf_counter();pairs=np.array([total_percolates_with_length(edges,reference) for edges in restored]);serial=time.perf_counter()-start
            np.testing.assert_array_equal(pairs[:,0],y[:48]);np.testing.assert_array_equal(pairs[:,1],lengths[:48])
            subset_times=[]
            for _ in range(3):
                start=time.perf_counter();probabilities(model,X[:48],L,checkpoint['calibration'],threads=1);subset_times.append(time.perf_counter()-start)
            report['benchmark']=dict(n=48,solver_serial_seconds=serial,cnn_serial_batch_seconds=subset_times,
                speedup=float(serial/np.median(subset_times)),full_ensemble_cnn_seconds=timings,
                note='CPU single thread, same 48 examples, two per p. CNN includes encoding; excludes warmup/loading/sampling/I/O. Solver input reconstruction excluded. One solver repeat; three CNN repeats.')
        (OUT/f'L{L}_metrics.json').write_text(json.dumps(report,indent=2)+'\n')
        np.savez_compressed(OUT/f'L{L}_predictions.npz',y=y,p=p,indices=ids,**probs)
        print(json.dumps({k:report[k] for k in ['L','n','role','metrics','curve_mae','signed_bias','signed_bias_bootstrap_95']},indent=2),flush=True)
        reports.append(report)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==frozen_hash
    (OUT/'summary.json').write_text(json.dumps(reports,indent=2)+'\n')
    plot(reports[-1])

def plot(report):
    os.environ.setdefault('MPLCONFIGDIR',str(PROJECT/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=report['per_p'];p=np.array([r['p'] for r in rows]);y=np.array([r['solver'] for r in rows]);ci=np.array([r['solver_wilson_95'] for r in rows])
    colors=dict(cnn='#2171b5',old_l16_cnn='#d95f0e',density_size='#238b45')
    labels=dict(cnn='Multi-size CNN (calibrated)',old_l16_cnn='Original L=16 CNN',density_size='Density + size baseline')
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    ax=axes[0,0];ax.errorbar(p,y,yerr=[np.maximum(0,y-ci[:,0]),np.maximum(0,ci[:,1]-y)],fmt='o',color='black',markersize=3,capsize=2,label='Solver: 50 realizations/p')
    for name,color in colors.items():ax.plot(p,[r[name]['probability'] for r in rows],color=color,label=labels[name])
    ax.set(xlabel='p',ylabel='Percolation probability',ylim=(-.02,1.02),title='L=36: final unseen-size test');ax.legend(fontsize=8)
    ax=axes[0,1];ax.axhline(0,color='black',linewidth=1)
    for name,color in colors.items():
        ax.plot(p,[r[name]['bias'] for r in rows],color=color,label=labels[name])
        bounds=np.array([r[name]['paired_bootstrap_bias_95'] for r in rows]);ax.fill_between(p,bounds[:,0],bounds[:,1],alpha=.12,color=color)
    ax.set(xlabel='p',ylabel='Prediction minus solver fraction',title='95% paired bootstrap; fixed models');ax.legend(fontsize=8)
    history=json.loads((ROOT/'history.json').read_text());ax=axes[1,0]
    ax.plot([r['epoch'] for r in history],[r['validation_loss'] for r in history],label='Equal-size mean validation')
    ax.plot([r['epoch'] for r in history],[r['validation_by_size']['30'] for r in history],label='L=30 validation')
    checkpoint=torch.load(ROOT/'cnn.pt',map_location='cpu',weights_only=False)
    ax.axvline(checkpoint['best_epoch'],linestyle='--',color='gray',label='Selected epoch')
    ax.set(xlabel='Fine-tuning epoch',ylabel='Binary cross-entropy',title='Selection uses validation only');ax.legend(fontsize=8)
    ax=axes[1,1];ax.plot([0,1],[0,1],'--',color='gray')
    with np.load(OUT/'L36_predictions.npz',allow_pickle=False) as d:
        for name,color in colors.items():
            q=d[name];xs=[];ys=[]
            for i in range(10):
                mask=(q>=i/10)&(q<(i+1)/10 if i<9 else q<=1)
                if mask.sum()>=10:xs.append(q[mask].mean());ys.append(d['y'][mask].mean())
            ax.plot(xs,ys,'o-',color=color,label=labels[name],markersize=4)
    ax.set(xlabel='Mean predicted probability in bin',ylabel='Observed positive fraction',xlim=(0,1),ylim=(0,1),title='L=36 calibration; bins with ≥10 examples');ax.legend(fontsize=8)
    for ax in axes.flat:ax.grid(alpha=.2)
    path=PROJECT/'figures/multisize_L36_evaluation.png';fig.savefig(path,dpi=170);plt.close(fig);print(path,flush=True)

if __name__=='__main__':main()
