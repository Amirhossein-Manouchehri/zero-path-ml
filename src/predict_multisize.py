"""Apply the saved multi-size CNN to physical ±1 bonds without running the solver."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy.special import expit
import torch
from src.dataset import PROJECT
from src.lattice_cnn import lattice_channels
from src.multisize_cnn import MultiSizeCNN

def load(path=PROJECT/'models/multisize_cnn_v1/cnn.pt'):
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    model=MultiSizeCNN().eval();model.load_state_dict(checkpoint['state_dict'])
    return model,checkpoint

def predict(X,L,model,calibration):
    torch.set_num_threads(2)
    channels=lattice_channels(X,L)
    with torch.no_grad():z=torch.cat([model(batch) for batch in channels.split(128)]).numpy()
    return expit(z/calibration['temperature']+calibration['intercept']+calibration['log_size_slope']*math.log(L/16))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--model',type=Path,default=PROJECT/'models/multisize_cnn_v1/cnn.pt')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.output and args.output.exists():parser.error('Choose a new output file; existing predictions are not overwritten')
    model,checkpoint=load(args.model)
    with np.load(args.dataset,allow_pickle=False) as data:
        meta=json.loads(data['metadata'].item());L=meta['L']
        q=predict(data['X'],L,model,checkpoint['calibration'])
        p=data['p_sample'] if 'p_sample' in data.files else None
    print(f'L={L}: {len(q)} predictions; training sizes={checkpoint["train_sizes"]}')
    if p is not None:
        for value in np.unique(p):
            mask=p==value;print(f'p={value:.6f}, N={int(mask.sum())}, mean probability={q[mask].mean():.6f}')
    else:print(f'Mean probability={q.mean():.6f}')
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('xb') as handle:
            payload=dict(probability=q,metadata=json.dumps(dict(L=L,model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest())))
            if p is not None:payload['p_sample']=p
            np.savez_compressed(handle,**payload)

if __name__=='__main__':main()
