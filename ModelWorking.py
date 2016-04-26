# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 13:49:49 2016

@author: kshain
"""

from qubitparameterestimation import model
from qubitparameterestimation import myio

test = myio.loadData('singleSeries.csv')
c = model.getCoefs(test)
model.posterior(c,50,70,200)