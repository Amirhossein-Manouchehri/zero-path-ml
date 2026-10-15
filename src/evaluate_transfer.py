"""Evaluate the frozen L=16 CNN and its density baseline on independent sizes."""
import hashlib
import json
import os
import time
import joblib
import numpy as np
import torch
from src.dataset import PROJECT,DEFAULT_SOLVER,load_solver,unpack_bonds,total_percolates_with_length
from src.lattice_cnn import lattice_channels
from src.predict_lattice import load_model
from src.train_lattice import metrics

MODEL=PROJECT/'models/L16_cnn_v1/cnn.pt'
OUT=PROJECT/'models/L16_transfer_v1'

def predict_fixed(model,X,L,temperature,threads=4):
    torch.set_num_threads(threads)
    # Include representation construction in inference timings.
    channels=lattice_channels(X,L)
    with torch.no_grad():
        return torch.cat([torch.sigmoid(model(batch)/temperature) for batch in channels.split(256)]).numpy()

def wilson(successes,n):
    q=successes/n;z=1.959963984540054;denom=1+z*z/n
    center=(q+z*z/(2*n))/denom
    half=z*np.sqrt(q*(1-q)/n+z*z/(4*n*n))/denom
    return [center-half,center+half]

def evaluate(L,model,temperature,density_model):
    path=PROJECT/f'data/raw/transfer_L{L}_n200_seed20261020.npz'
    with np.load(path,allow_pickle=False) as data:
        X,y,p,lengths,seconds=(data[k] for k in ['X','y','p_sample','path_length','solver_seconds'])
        meta=json.loads(data['metadata'].item())
    assert len(y)==meta['samples']==meta['target_samples']==4800
    assert X.shape==(4800,L*(2*L-1)) and np.isin(X,[-1,1]).all() and np.isin(y,[0,1]).all()
    assert len(np.unique(X,axis=0))==len(X)
    assert all((p==value).sum()==200 for value in np.unique(p))
    assert np.array_equal(lengths==0,y==0)
    assert np.all(lengths[y==1]%2==0) and np.all(lengths<=L*L//2)
    assert meta['cnn_sha256']==hashlib.sha256(MODEL.read_bytes()).hexdigest()
    assert meta['solver_sha256']==hashlib.sha256(DEFAULT_SOLVER.read_bytes()).hexdigest()
    predict_fixed(model,X[:8],L,temperature)  # Exclude initial warmup from timings.
    times=[]
    for _ in range(3):
        start=time.perf_counter();q=predict_fixed(model,X,L,temperature);times.append(time.perf_counter()-start)
    start=time.perf_counter()
    density_q=density_model.predict_proba((X==-1).mean(axis=1)[:,None])[:,1]
    density_seconds=time.perf_counter()-start
    per_p=[];rng=np.random.default_rng(2026+L)
    for value in np.unique(p):
        mask=p==value;n=int(mask.sum());labels=y[mask]
        row=dict(p=float(value),n=n,solver_probability=float(labels.mean()),solver_wilson_95=wilson(int(labels.sum()),n))
        for name,prob in [('cnn',q),('density',density_q)]:
            residual=prob[mask]-labels
            draws=rng.integers(n,size=(3000,n))
            bootstrap=residual[draws].mean(axis=1)
            row[name]=dict(probability=float(prob[mask].mean()),bias=float(residual.mean()),
                paired_bootstrap_bias_95=np.quantile(bootstrap,[.025,.975]).tolist())
        row['path_statistics']=dict(mean_unconditional_over_L2=float(lengths[mask].mean()/L**2),
            variance_unconditional_over_L2=float(lengths[mask].var()/L**2),
            mean_conditional=float(lengths[mask & (y==1)].mean()) if (mask & (y==1)).any() else None)
        per_p.append(row)
    # Fair one-core benchmark on 96 representative examples, four at every p.
    indices=np.arange(96)
    reference=load_solver(DEFAULT_SOLVER)
    restored=[unpack_bonds(row,L) for row in X[indices]]
    total_percolates_with_length(restored[0],reference)
    start=time.perf_counter()
    pairs=np.array([total_percolates_with_length(edges,reference) for edges in restored])
    solver_serial=time.perf_counter()-start
    np.testing.assert_array_equal(pairs[:,0],y[indices]);np.testing.assert_array_equal(pairs[:,1],lengths[indices])
    predict_fixed(model,X[indices[:8]],L,temperature,threads=1)
    single_times=[]
    for _ in range(3):
        start=time.perf_counter();serial_q=predict_fixed(model,X[indices],L,temperature,threads=1)
        single_times.append(time.perf_counter()-start)
    np.testing.assert_allclose(serial_q,q[indices],atol=2e-6,rtol=2e-5)
    score={name:metrics(y,prob) for name,prob in [('cnn',q),('density',density_q)]}
    for name,prob in [('cnn',q),('density',density_q)]:
        score[name]['accuracy_wilson_95']=wilson(int(((prob>=.5)==y).sum()),len(y))
    bias_rng=np.random.default_rng(183+L)
    means=[]
    for value in np.unique(p):
        residual=q[p==value]-y[p==value]
        means.append(residual[bias_rng.integers(len(residual),size=(3000,len(residual)))].mean(axis=1))
    overall_bias_interval=np.quantile(np.mean(means,axis=0),[.025,.975]).tolist()
    report=dict(L=L,trained_L=16,N=len(y),samples_per_p=200,positive=int(y.sum()),
        dataset=str(path),dataset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),metadata=meta,
        metrics=score,per_p=per_p,
        curve_mae={name:float(np.mean([abs(r[name]['bias']) for r in per_p])) for name in score},
        curve_signed_bias={name:float(np.mean([r[name]['bias'] for r in per_p])) for name in score},
        cnn_overall_signed_bias_stratified_bootstrap_95=overall_bias_interval,
        inference_seconds_4threads=times,inference_seconds_4threads_median=float(np.median(times)),
        density_inference_seconds=density_seconds,
        benchmark=dict(n=96,solver_serial_seconds=solver_serial,cnn_single_thread_seconds=single_times,
            speedup=float(solver_serial/np.median(single_times)),
            note='Same 96 examples; solver serial/Numba single thread; CNN one CPU thread in batches. CNN includes lattice encoding. Warmup, model loading, random-bond sampling and file I/O excluded. Three CNN repeats; one solver repeat.'),
        limitations='Independent p-stratified Monte Carlo ensemble; no transfer fitting or calibration. Intervals condition on a single frozen model. These are solver comparisons, not certified physical truth or fitted critical exponents.')
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/f'L{L}_metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(OUT/f'L{L}_predictions.npz',cnn=q,density=density_q,y=y,p=p,path_length=lengths)
    print(json.dumps({k:report[k] for k in ['L','N','positive','metrics','curve_mae','curve_signed_bias','benchmark']},indent=2),flush=True)
    return report

def plot(reports):
    os.environ.setdefault('MPLCONFIGDIR',str(PROJECT/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,len(reports),figsize=(12,8),layout='constrained',squeeze=False)
    for column,report in enumerate(reports):
        rows=report['per_p'];p=np.array([r['p'] for r in rows]);observed=np.array([r['solver_probability'] for r in rows])
        ci=np.array([r['solver_wilson_95'] for r in rows]);ax=axes[0,column]
        ax.errorbar(p,observed,yerr=[np.maximum(0,observed-ci[:,0]),np.maximum(0,ci[:,1]-observed)],fmt='o',color='black',markersize=3,capsize=2,label='Solver (200 realizations/p)')
        for name,color in [('cnn','#2171b5'),('density','#238b45')]:
            ax.plot(p,[r[name]['probability'] for r in rows],color=color,label=f'Frozen L=16 {name}')
        ax.set(title=f'L={report["L"]}: transfer without retraining',xlabel='p',ylabel='Mean percolation probability',ylim=(-.02,1.02))
        ax.legend(fontsize=8)
        ax=axes[1,column];ax.axhline(0,color='black',linewidth=1)
        for name,color in [('cnn','#2171b5'),('density','#238b45')]:
            ax.plot(p,[r[name]['bias'] for r in rows],color=color,label=name)
            bounds=np.array([r[name]['paired_bootstrap_bias_95'] for r in rows])
            ax.fill_between(p,bounds[:,0],bounds[:,1],alpha=.15,color=color)
        ax.set(xlabel='p',ylabel='Prediction minus solver fraction',title='95% paired bootstrap intervals; fixed model')
        ax.legend(fontsize=8)
    for ax in axes.flat:ax.grid(alpha=.2)
    path=PROJECT/'figures/L16_transfer_L20_L24.png'
    fig.savefig(path,dpi=170);plt.close(fig)
    print(f'Saved {path}',flush=True)

def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--L',type=int,nargs='+',default=[20,24])
    args=parser.parse_args()
    original_hash=hashlib.sha256(MODEL.read_bytes()).hexdigest()
    model,checkpoint=load_model(MODEL)
    density=joblib.load(PROJECT/'models/L16_cnn_v1/baselines.joblib')['density']
    reports=[evaluate(L,model,checkpoint['temperature'],density) for L in args.L]
    assert hashlib.sha256(MODEL.read_bytes()).hexdigest()==original_hash
    (OUT/'summary.json').write_text(json.dumps(reports,indent=2)+'\n')
    plot(reports)

if __name__=='__main__':main()
