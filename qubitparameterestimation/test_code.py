import unittest

from qubitparameterestimation import model
import numpy as np
from qubitparameterestimation import myio



class TestMeasurement(unittest.TestCase):
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

class TestPosteriorFromCoefs(unittest.TestCase):
    def test_posterior(self):
        filename = 'singleSeries.csv'
        testdata = myio.loadData(filename)
        testcoefs = model.getCoefs(testdata)
        posteriorMax = model.posterior(testcoefs,50,70,50)
        self.assertTrue(posteriorMax>57 and posteriorMax<58)

class TestSimulatedCoefsAndExpectedBz(unittest.TestCase):
    def test_simulatedcoefs(self):
        BzDiffused = model.randomWalk(60, 6.7)
        c = model.simulateCoefs(.25, .67, 0, 25, BzDiffused)
        Bz = model.expectedBz(c[99,:])
        self.assertTrue(Bz>59 and Bz<61)
        
class TestOptimizedCoefs(unittest.TestCase):
    def test_optimizedcoefs(self):
        BzDiffused = model.randomWalk(70, 6.7)
        c = model.optimizedCoefs(2, .25, .67, 5, 6.7, BzDiffused)
        diffusionVector = model.diffusionVector(6.7, c.shape[1])
        expvar = model.expVar(c[1,:],10,diffusionVector,.25,.67)
        self.assertTrue(expvar>100 and expvar<200)

class TestMSE(unittest.TestCase):
    def test_mse(self):
        BzDiffused = model.randomWalk(70, 6.7)
        c = np.empty((2,2,10000))
        c[0] = model.optimizedCoefs(2, .25, .67, 5, 6.7, BzDiffused)
        c[1] = model.optimizedCoefs(2, .25, .67, 5, 6.7, BzDiffused)
        mse = model.getMSE(c, BzDiffused)
        self.assertTrue(mse[1]>10 and mse[1]<70)


        
if __name__ == '__main__':
    unittest.main()
    