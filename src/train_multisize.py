"""Fine-tune the size-aware CNN on five sizes; use L=30 only for validation."""
import copy
import hashlib
import json
import math
import time
import joblib
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
import torch
from src.dataset import PROJECT
from src.lattice_cnn import lattice_channels
from src.multisize_cnn import MultiSizeCNN
from src.train_lattice import make_splits

OUT=PROJECT/'models/multisize_cnn_v1'
TRAIN_SIZES=[16,20,24,28,32]

def fit_calibration(z,y,s,w):
    def objective(theta):
        calibrated=z/np.exp(theta[0])+theta[1]+theta[2]*s
        return float(np.average(np.logaddexp(0,calibrated)-y*calibrated,weights=w))
    result=minimize(objective,[0,0,0],method='L-BFGS-B',bounds=[(-2,2),(-3,3),(-5,5)])
    if not result.success:raise RuntimeError(result.message)
    return dict(temperature=float(np.exp(result.x[0])),intercept=float(result.x[1]),log_size_slope=float(result.x[2]))

def source_path(L):
    if L==16:return PROJECT/'data/raw/training_L16_n500_seed20261015.npz'
    if L in (20,24):return PROJECT/f'data/raw/transfer_L{L}_n200_seed20261020.npz'
    return PROJECT/f'data/raw/multisize_L{L}_n50_seed20261028.npz'

def read_data(L):
    path=source_path(L)
    with np.load(path,allow_pickle=False) as data:
        X,y,p=data['X'],data['y'],data['p_sample']
        meta=json.loads(data['metadata'].item())
    assert len(y)==meta['samples']==meta['target_samples']
    assert X.shape==(len(y),L*(2*L-1)) and np.isin(y,[0,1]).all()
    assert len(np.unique(X,axis=0))==len(X)
    if L==16:
        with np.load(PROJECT/'models/L16_cnn_v1/split.npz',allow_pickle=False) as split:
            parts=[split[name] for name in ['train','validation','test']]
    elif L==30:parts=[np.array([],int),np.arange(len(y)),np.array([],int)]
    else:parts=make_splits(y,p,seed=52)
    assert len(np.unique(np.concatenate(parts)))==len(y)
    return dict(X=X,y=y,p=p,channels=lattice_channels(X,L),train=parts[0],val=parts[1],test=parts[2],
        metadata=meta,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def logits(model,channels):
    model.eval()
    with torch.no_grad():return torch.cat([model(batch) for batch in channels.split(128)]).numpy()

def validation(model,data):
    result={}
    for L,values in data.items():
        ids=values['val'];z=logits(model,values['channels'][ids]);y=values['y'][ids]
        result[L]=float(np.mean(np.logaddexp(0,z)-y*z))
    return float(np.mean(list(result.values()))),result

def main():
    torch.set_num_threads(2);torch.manual_seed(42);torch.use_deterministic_algorithms(True)
    rng=np.random.default_rng(42)
    OUT.mkdir(parents=True,exist_ok=False)
    data={L:read_data(L) for L in TRAIN_SIZES+[30]}
    split={f'L{L}_{part}':values[part] for L,values in data.items() for part in ['train','val','test']}
    np.savez_compressed(OUT/'split_indices.npz',**split)
    manifest={L:dict(path=d['path'],sha256=d['sha256'],train=len(d['train']),validation=len(d['val']),diagnostic_test=len(d['test'])) for L,d in data.items()}
    (OUT/'data_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    count=min(len(data[L]['train']) for L in TRAIN_SIZES)
    print(f'Equal size sampling: {count} examples/size/epoch; training sizes {TRAIN_SIZES}; L30 validation only',flush=True)
    # A smooth baseline using actual negative-bond density and log(size), never p.
    baseline_X=[];baseline_y=[]
    for L in TRAIN_SIZES:
        d=data[L];ids=rng.choice(d['train'],count,replace=False)
        density=(d['X'][ids]==-1).mean(1)
        baseline_X.append(np.column_stack([density,np.full(count,math.log(L/16))]))
        baseline_y.append(d['y'][ids])
    baseline=make_pipeline(StandardScaler(),PolynomialFeatures(2,include_bias=False),LogisticRegression(C=1,max_iter=2000))
    baseline.fit(np.concatenate(baseline_X),np.concatenate(baseline_y))
    joblib.dump(baseline,OUT/'density_size_baseline.joblib')
    original=PROJECT/'models/L16_cnn_v1/cnn.pt'
    original_hash=hashlib.sha256(original.read_bytes()).hexdigest()
    checkpoint=torch.load(original,map_location='cpu',weights_only=False)
    model=MultiSizeCNN();model.initialize_from_l16(checkpoint['state_dict'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    loss_function=torch.nn.BCEWithLogitsLoss()
    best,initial=validation(model,data);best_state=copy.deepcopy(model.state_dict());best_epoch=0;stale=0
    history=[dict(epoch=0,validation_loss=best,validation_by_size=initial)]
    start=time.perf_counter()
    for epoch in range(1,21):
        batches=[]
        for L in TRAIN_SIZES:
            ids=rng.choice(data[L]['train'],count,replace=False)
            for offset in range(0,count,64):batches.append((L,ids[offset:offset+64]))
        rng.shuffle(batches);model.train();total=0;seen=0
        for L,ids in batches:
            d=data[L];optimizer.zero_grad()
            loss=loss_function(model(d['channels'][ids]),torch.tensor(d['y'][ids],dtype=torch.float32))
            loss.backward();optimizer.step();total+=loss.item()*len(ids);seen+=len(ids)
        score,by_size=validation(model,data)
        row=dict(epoch=epoch,train_loss=total/seen,validation_loss=score,validation_by_size=by_size)
        history.append(row)
        print(f'Epoch {epoch}: train={row["train_loss"]:.4f}, equal-size validation={score:.4f}, L30={by_size[30]:.4f}, elapsed={time.perf_counter()-start:.1f}s',flush=True)
        if score<best-1e-4:best=score;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch;stale=0
        else:stale+=1
        torch.save(dict(state_dict=best_state,best_epoch=best_epoch,history=history),OUT/'best_uncalibrated.pt')
        (OUT/'history.json').write_text(json.dumps(history,indent=2)+'\n')
        if stale>=5:break
    model.load_state_dict(best_state)
    zs=[];ys=[];sizes=[];weights=[];baseline_z=[]
    for L,d in data.items():
        ids=d['val'];zs.append(logits(model,d['channels'][ids]));ys.append(d['y'][ids])
        sizes.append(np.full(len(ids),math.log(L/16)));weights.append(np.full(len(ids),1/len(ids)))
        baseline_z.append(baseline.decision_function(np.column_stack([(d['X'][ids]==-1).mean(1),sizes[-1]])))
    z=np.concatenate(zs).astype(float);y=np.concatenate(ys);s=np.concatenate(sizes);w=np.concatenate(weights)
    calibration=fit_calibration(z,y,s,w)
    baseline_calibration=fit_calibration(np.concatenate(baseline_z),y,s,w)
    payload=dict(state_dict=best_state,architecture='MultiSizeCNN v1',train_sizes=TRAIN_SIZES,
        size_validation=30,calibration=calibration,baseline_calibration=baseline_calibration,best_epoch=best_epoch,seed=42,
        warm_start_sha256=original_hash,data_manifest=manifest,torch_version=torch.__version__)
    torch.save(payload,OUT/'cnn.pt.tmp')
    (OUT/'cnn.pt.tmp').replace(OUT/'cnn.pt')
    (OUT/'training_summary.json').write_text(json.dumps(dict(best_epoch=best_epoch,calibration=calibration,baseline_calibration=baseline_calibration,
        training_seconds=time.perf_counter()-start,examples_per_size_per_epoch=count,parameters=sum(v.numel() for v in model.parameters()),
        train_sizes=TRAIN_SIZES,size_validation=30,data_manifest=manifest),indent=2)+'\n')
    assert hashlib.sha256(original.read_bytes()).hexdigest()==original_hash
    print(f'Frozen checkpoint saved at {OUT}/cnn.pt, best epoch={best_epoch}, calibration={calibration}',flush=True)

if __name__=='__main__':main()
