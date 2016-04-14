import unittest

import model
import numpy as np



class TestModel(unittest.TestCase):
    def test_outputsort(self):
        Bz = 60
        t = 100
        k = 50
        BzDiffused = model.randomWalk(Bz)
        output = model.measurementDriftDiffusion(BzDiffused,t,k)
        self.assertTrue(output in [-1,1])
        
        
if __name__ == '__main__':
    unittest.main()
    