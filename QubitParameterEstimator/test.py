from unittest import TestCase

import model
import numpy as np

class TestModel(TestCase):
    def outputSorT(self):
        Bz = 60
        t = 100
        k = 50
        BzDiffused = model.randomWalk(Bz)
        output = model.measurementDriftDiffusion(BzDiffused,t,k)
        self.assertTrue(output in [-1,1])
