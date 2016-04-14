import unittest

from qubitparameterestimation import model
import numpy as np



class TestModel(unittest.TestCase):
    def test_outputsort(self):
        Bz = 60
        t = 100
        k = 50
        BzDiffused = model.randomWalk(Bz)
        output = model.measurementDriftDiffusion(BzDiffused,t,25,k)
        self.assertTrue(output in [-1,1])
        
        
if __name__ == '__main__':
    unittest.main()
    