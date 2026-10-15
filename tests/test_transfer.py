import unittest
from unittest.mock import patch
import numpy as np
from src.dataset import DEFAULT_SOLVER,load_solver,generate_bonds,total_percolates,total_percolates_with_length

@unittest.skipUnless(DEFAULT_SOLVER.is_file(), "Optional reference research solver is not installed")
class TransferTests(unittest.TestCase):
    def test_preserved_labels_and_reference_lengths(self):
        solver=load_solver(DEFAULT_SOLVER)
        rng=np.random.default_rng(8462)
        labels=[]
        for p in [0,.1,.3,.5,1]:
            edges=generate_bonds(4,p,rng,solver)
            actual=total_percolates_with_length(edges,solver)
            with patch.object(solver.np.random,'choice',return_value=edges.copy()):
                expected=solver.run_single_trial_no_pbc(4,p)
            self.assertEqual(actual,expected)
            self.assertEqual(total_percolates(edges,solver),actual[0])
            self.assertEqual(actual[1]==0,actual[0]==0)
            if actual[0]:
                self.assertEqual(actual[1]%2,0)
                self.assertLessEqual(actual[1],8)
            labels.append(actual[0])
        self.assertIn(0,labels);self.assertIn(1,labels)

if __name__=='__main__': unittest.main()
