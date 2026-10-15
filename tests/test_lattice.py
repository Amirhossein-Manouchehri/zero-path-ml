import unittest
import numpy as np
import torch
from src.lattice_cnn import CylindricalConv,LatticeCNN,lattice_channels
from src.train_lattice import make_splits

class LatticeTests(unittest.TestCase):
    def test_padding_wraps_only_horizontally(self):
        layer=CylindricalConv(1,1,1)
        x=torch.zeros(1,1,16,16); x[0,0,0,-1]=7
        with torch.no_grad():
            layer.conv.weight.zero_(); layer.conv.bias.zero_()
            layer.conv.weight[0,0,1,0]=1
            self.assertEqual(layer(x)[0,0,0,0].item(),7)
            layer.conv.weight.zero_(); layer.conv.weight[0,0,0,1]=1
            self.assertEqual(layer(x)[0,0,0,-1].item(),0)
    def test_bonds_masks_and_variable_size(self):
        model=LatticeCNN().eval()
        for L in (16,36):
            X=np.ones((2,L*(2*L-1)),dtype=np.int8); X[0,-1]=-1
            c=lattice_channels(X,L)
            self.assertEqual(c[0,1,L-2,-1],-1)
            self.assertTrue(torch.all(c[:,1:3,-1]==0))
            self.assertEqual(c[:,3].sum().item(),2*L)
            with torch.no_grad(): self.assertEqual(model(c).shape,(2,))
    def test_saved_split_is_disjoint_and_balanced_per_p(self):
        p=np.repeat([.08,.10,.12],500)
        y=np.tile(np.arange(500)%2,3)
        parts=make_splits(y,p)
        self.assertEqual([len(v) for v in parts],[1050,225,225])
        self.assertEqual(len(np.unique(np.concatenate(parts))),1500)
        for part,count in zip(parts,[350,75,75]):
            for value in np.unique(p): self.assertEqual(int((p[part]==value).sum()),count)
        for a,b in zip(parts,make_splits(y,p)): np.testing.assert_array_equal(a,b)

if __name__=='__main__': unittest.main()
