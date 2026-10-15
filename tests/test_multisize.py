import unittest
import numpy as np
import torch
from scipy.special import expit
from src.lattice_cnn import LatticeCNN,lattice_channels
from src.multisize_cnn import MultiSizeCNN
from src.train_multisize import fit_calibration

class MultiSizeTests(unittest.TestCase):
    def test_warm_start_preserves_initial_predictions(self):
        torch.set_num_threads(2);torch.manual_seed(18)
        old=LatticeCNN().eval();new=MultiSizeCNN().eval()
        new.initialize_from_l16(old.state_dict())
        for L in [16,20,32,36]:
            X=np.random.default_rng(L).choice([-1,1],size=(3,L*(2*L-1))).astype(np.int8)
            channels=lattice_channels(X,L)
            with torch.no_grad():torch.testing.assert_close(old(channels),new(channels),atol=2e-6,rtol=1e-5)
    def test_calibration_corrects_known_size_bias(self):
        rng=np.random.default_rng(571)
        z=rng.normal(0,1,2000);s=np.repeat([0,.4,.7,.8],500)
        y=rng.binomial(1,expit(z-.5+.8*s))
        c=fit_calibration(z,y,s,np.ones(len(z)))
        adjusted=z/c['temperature']+c['intercept']+c['log_size_slope']*s
        raw=np.mean(np.logaddexp(0,z)-y*z)
        fitted=np.mean(np.logaddexp(0,adjusted)-y*adjusted)
        self.assertLess(fitted,raw)
        self.assertGreater(c['log_size_slope'],0)

if __name__=='__main__':unittest.main()
