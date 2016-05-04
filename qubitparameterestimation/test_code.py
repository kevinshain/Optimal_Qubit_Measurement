import unittest

from qubitparameterestimation import model
import numpy as np
from qubitparameterestimation import myio



class TestModel(unittest.TestCase):
    def test_outputsort(self):
        Bz = 60
        t = 100
        k = 50
        diffusion = 6.7
        BzDiffused = model.randomWalk(Bz,diffusion)
        output = model.measurementDriftDiffusion(BzDiffused,t,25,k)
        self.assertTrue(output in [-1,1])

class TestIO(unittest.TestCase):
    def test_io(self):
        filename = 'singleSeries.csv'
        testdata = myio.loadData(filename)
        self.assertTrue(issubclass(type(testdata), np.ndarray))

class TestLikelihood(unittest.TestCase):
    def test_likelihood(self):
        testdata = myio.loadData('singleSeries.csv')
        Bz = np.linspace(50,70,21)
        lk = np.empty(len(Bz))
        for i in range(len(Bz)):
        	lk[i] = model.likelihood(testdata,Bz[i])
        self.assertTrue(np.argmax(lk)>6 and np.argmax(lk)<16)
        
        
if __name__ == '__main__':
    unittest.main()
    