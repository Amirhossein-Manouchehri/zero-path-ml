"""Train once with fixed splits, validation-only early stopping/calibration, and final test evaluation."""
import argparse
import copy
import hashlib
import json
import os
import time

import joblib
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, confusion_matrix, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader,TensorDataset

from src.dataset import PROJECT
from src.lattice_cnn import LatticeCNN,lattice_channels

def make_splits(y,p,seed=42):
    splits = [[],[],[]]
    for value in np.unique(p):
        ids = np.flatnonzero(p==value)
        def stratify(ids):
            counts = np.bincount(y[ids],minlength=2)
            return y[ids] if counts.min() >= 2 else None
        train,rest = train_test_split(ids,test_size=.30,random_state=seed,stratify=stratify(ids))
        val,test = train_test_split(rest,test_size=.50,random_state=seed+1,stratify=stratify(rest))
        for out,part in zip(splits,[train,val,test]): out.extend(part)
    return [np.array(part,dtype=int) for part in splits]

def metrics(y,q):
    return dict(accuracy=float(accuracy_score(y,q>=.5)),
        balanced_accuracy=float(balanced_accuracy_score(y,q>=.5)),
        roc_auc=float(roc_auc_score(y,q)), brier=float(brier_score_loss(y,q)),
        log_loss=float(log_loss(y,q,labels=[0,1])),
        confusion_matrix=confusion_matrix(y,q>=.5,labels=[0,1]).tolist())

def predict_logits(model,x):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(batch) for batch in x.split(256)]).numpy()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',default=str(PROJECT/'data/raw/training_L16_n500_seed20261015.npz'))
    parser.add_argument('--epochs',type=int,default=40)
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.use_deterministic_algorithms(True)
    output=PROJECT/'models/L16_cnn_v1'
    output.mkdir(parents=True,exist_ok=False)
    with np.load(args.dataset,allow_pickle=False) as data:
        X,y,p=data['X'],data['y'],data['p_sample']
        meta=json.loads(data['metadata'].item())
    assert len(y)==meta['target_samples']==meta['samples']
    assert np.isin(y,[0,1]).all() and len(np.unique(X,axis=0))==len(X)
    train,val,test=make_splits(y,p)
    assert len(np.unique(np.concatenate([train,val,test])))==len(X)
    np.savez_compressed(output/'split.npz',train=train,validation=val,test=test,
        dataset_sha256=hashlib.sha256(open(args.dataset,'rb').read()).hexdigest())
    print(f'Fixed split: train={len(train)}, validation={len(val)}, test={len(test)}',flush=True)
    # All preprocessing is fit only to training rows. p is never a model input.
    logistic=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,C=1.0))
    density_model=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=2000))
    density=(X==-1).mean(axis=1)[:,None]
    logistic.fit(X[train],y[train])
    density_model.fit(density[train],y[train])
    joblib.dump(dict(logistic=logistic,density=density_model),output/'baselines.joblib')
    channels=lattice_channels(X,meta['L'])
    model=LatticeCNN()
    optim=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    loss_function=torch.nn.BCEWithLogitsLoss()
    loader=DataLoader(TensorDataset(channels[train],torch.tensor(y[train],dtype=torch.float32)),
        batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(42))
    best=float('inf'); stale=0; best_epoch=0; best_state=None; history=[]
    start=time.perf_counter()
    for epoch in range(1,args.epochs+1):
        model.train(); total=0
        for inputs,targets in loader:
            optim.zero_grad(); loss=loss_function(model(inputs),targets)
            loss.backward(); optim.step(); total+=loss.item()*len(inputs)
        logits=predict_logits(model,channels[val])
        val_loss=float(np.mean(np.logaddexp(0,logits)-y[val]*logits))
        row=dict(epoch=epoch,train_loss=total/len(train),validation_loss=val_loss)
        history.append(row)
        print(f'Epoch {epoch}: train={row["train_loss"]:.4f}, validation={val_loss:.4f}, elapsed={time.perf_counter()-start:.1f}s',flush=True)
        if val_loss < best-1e-4:
            best=val_loss; best_state=copy.deepcopy(model.state_dict()); best_epoch=epoch; stale=0
            torch.save(dict(state_dict=best_state,best_epoch=best_epoch,metadata=meta),output/'best_uncalibrated.pt')
        else: stale+=1
        (output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
        if stale>=8: break
    model.load_state_dict(best_state)
    val_logits=predict_logits(model,channels[val])
    # Fit one temperature only on validation; leave test labels untouched until here.
    def objective(log_temp):
        z=val_logits/np.exp(log_temp)
        return float(np.mean(np.logaddexp(0,z)-y[val]*z))
    calibration=minimize_scalar(objective,bounds=(-2,2),method='bounded')
    temperature=float(np.exp(calibration.x))
    torch.save(dict(state_dict=best_state,temperature=temperature,best_epoch=best_epoch,
        metadata=meta,torch_version=torch.__version__,architecture='LatticeCNN v1'),output/'cnn.pt')
    test_logits=predict_logits(model,channels[test])
    predictions=dict(
        constant=np.full(len(test),float(y[train].mean())),
        density=density_model.predict_proba(density[test])[:,1],
        logistic=logistic.predict_proba(X[test])[:,1],
        cnn_uncalibrated=expit(test_logits), cnn=expit(test_logits/temperature))
    scores={name:metrics(y[test],q) for name,q in predictions.items()}
    per_p=[]
    rng=np.random.default_rng(812)
    for value in np.unique(p[test]):
        mask=p[test]==value; labels=y[test][mask]; n=len(labels); fraction=float(labels.mean())
        z=1.959963984540054; denom=1+z*z/n
        center=(fraction+z*z/(2*n))/denom
        half=z*np.sqrt(fraction*(1-fraction)/n+z*z/(4*n*n))/denom
        row=dict(p=float(value),n=n,solver=fraction,solver_wilson_95=[center-half,center+half])
        for name,q in predictions.items():
            residual=q[mask]-labels
            bootstrap=residual[rng.integers(n,size=(2000,n))].mean(axis=1)
            row[name]=dict(mean=float(q[mask].mean()),difference=float(residual.mean()),
                paired_bootstrap_difference_95=np.quantile(bootstrap,[.025,.975]).tolist())
        per_p.append(row)
    curve_mae={name:float(np.mean([abs(row[name]['difference']) for row in per_p])) for name in predictions}
    report=dict(dataset=args.dataset,dataset_metadata=meta,seed=42,
        split_counts=dict(train=len(train),validation=len(val),test=len(test)),
        split_method='Within each p, 70/15/15; stratify by label when possible; unique configurations required',
        class_counts={name:np.bincount(y[ids],minlength=2).tolist() for name,ids in [('train',train),('validation',val),('test',test)]},
        best_epoch=best_epoch,temperature=temperature,training_seconds=time.perf_counter()-start,
        test_metrics=scores,curve_mae=curve_mae,per_p=per_p,
        parameters=sum(v.numel() for v in model.parameters()),torch_version=torch.__version__,
        limitations='L=16 only; 75 test realizations per p; bootstrap is conditional on fixed trained model and approximate with label-stratified split; no cross-size transfer tested')
    (output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(output/'test_predictions.npz',indices=test,y=y[test],p=p[test],**predictions)
    print(json.dumps(dict(best_epoch=best_epoch,temperature=temperature,test_metrics=scores,curve_mae=curve_mae),indent=2),flush=True)
    print(f'Saved {output}',flush=True)

if __name__=='__main__': main()
