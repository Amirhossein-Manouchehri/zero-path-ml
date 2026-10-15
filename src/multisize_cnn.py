"""Size-conditioned extension of the L=16 CNN; no sampling p is supplied."""
import math
import torch
from torch import nn
from src.lattice_cnn import LatticeCNN

class MultiSizeCNN(LatticeCNN):
    def __init__(self):
        super().__init__()
        self.head=nn.Sequential(nn.Linear(34,32),nn.ReLU(),nn.Linear(32,1))

    def initialize_from_l16(self,state):
        self.features.load_state_dict({k[len('features.'):]:v for k,v in state.items() if k.startswith('features.')})
        with torch.no_grad():
            self.head[0].weight[:,:33].copy_(state['head.0.weight'])
            self.head[0].weight[:,33].zero_()
            self.head[0].bias.copy_(state['head.0.bias'])
            self.head[2].weight.copy_(state['head.2.weight'])
            self.head[2].bias.copy_(state['head.2.bias'])

    def forward(self,x):
        negative=((1-x[:,0])/2).sum((1,2))+(((1-x[:,1])/2)*x[:,2]).sum((1,2))
        density=negative/(x.shape[2]*x.shape[3]+x[:,2].sum((1,2)))
        features=self.features(x)
        size=x.new_full((len(x),1),math.log(x.shape[2]/16))
        pooled=torch.cat([features.mean((2,3)),features.amax((2,3)),density[:,None],size],1)
        return self.head(pooled).squeeze(1)
