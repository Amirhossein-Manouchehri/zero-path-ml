"""A size-flexible lattice CNN with periodic horizontal and open vertical padding."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def lattice_channels(X, L):
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != L*(2*L-1) or not np.isin(X,[-1,1]).all():
        raise ValueError('Expected physical ±1 bonds in the recorded feature order')
    out = np.zeros((len(X),5,L,L), dtype=np.float32)
    out[:,0] = X[:,:L*L].reshape(-1,L,L)
    out[:,1,:-1] = X[:,L*L:].reshape(-1,L-1,L)
    out[:,2,:-1] = 1  # Valid vertical-bond mask (zero means no bond).
    out[:,3,0] = 1    # Top boundary.
    out[:,4,-1] = 1   # Bottom boundary.
    return torch.from_numpy(out)

class CylindricalConv(nn.Module):
    def __init__(self, inputs, outputs, dilation):
        super().__init__()
        self.dilation = dilation
        self.conv = nn.Conv2d(inputs,outputs,3,dilation=dilation)
    def forward(self,x):
        d = self.dilation
        x = F.pad(x,(d,d,0,0),mode='circular')
        x = F.pad(x,(0,0,d,d),mode='constant',value=0)
        return self.conv(x)

class LatticeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for i,d in enumerate([1,2,4,8,4,1]):
            layers += [CylindricalConv(5 if i == 0 else 16,16,d),nn.ReLU()]
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(33,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,x):
        # Fraction of negative physical bonds: global baseline plus learned spatial features.
        negatives = ((1-x[:,0])/2).sum((1,2)) + (((1-x[:,1])/2)*x[:,2]).sum((1,2))
        density = negatives/(x.shape[2]*x.shape[3]+x[:,2].sum((1,2)))
        features = self.features(x)
        pooled = torch.cat([features.mean((2,3)),features.amax((2,3)),density[:,None]],dim=1)
        return self.head(pooled).squeeze(1)
