"""Predict solver percolation probabilities from a saved physical bond matrix."""
import argparse
from pathlib import Path
import numpy as np
import torch
from src.dataset import PROJECT
from src.lattice_cnn import LatticeCNN,lattice_channels

def load_model(path=PROJECT/'models/L16_cnn_v1/cnn.pt'):
    # Only load checkpoints created locally by this project.
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    model=LatticeCNN().eval()
    model.load_state_dict(checkpoint['state_dict'])
    return model,checkpoint

def predict(X,L,model,temperature=1):
    torch.set_num_threads(4)
    inputs=lattice_channels(X,L)
    with torch.no_grad():
        return torch.cat([torch.sigmoid(model(part)/temperature) for part in inputs.split(256)]).numpy()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--model',type=Path,default=PROJECT/'models/L16_cnn_v1/cnn.pt')
    args=parser.parse_args()
    model,checkpoint=load_model(args.model)
    import json
    with np.load(args.dataset,allow_pickle=False) as data:
        meta=json.loads(data['metadata'].item())
        q=predict(data['X'],meta['L'],model,checkpoint['temperature'])
    print(f'{len(q)} predictions; ensemble mean={q.mean():.6f}')
    if meta['L'] != checkpoint['metadata']['L']:
        print('This lattice size was not used in training; cross-size accuracy is unvalidated.')

if __name__=='__main__': main()
