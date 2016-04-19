import numpy as np
import csv


class MetaSeries(np.ndarray):
    """Array with metadata."""

    def __new__(cls, array, dtype=None, order=None, **kwargs):
        obj = np.asarray(array, dtype=dtype, order=order).view(cls)                                 
        obj.device = kwargs.get('device',None)
        obj.alpha = kwargs.get('alpha',None)
        obj.beta = kwargs.get('beta',None)
        obj.drift = kwargs.get('drift',None)
        obj.diffusion = kwargs.get('diffusion',None)
        
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.device = getattr(obj, 'device', None)
        self.alpha = getattr(obj, 'alpha', None)
        self.beta = getattr(obj, 'beta', None)
        self.drift = getattr(obj, 'drift', None)
        self.diffusion = getattr(obj, 'diffusion', None)

def loadData(filename):
    data = np.genfromtxt(filename, delimiter=',', skip_header=1, usecols=(0,1,2))
    
    with open(filename) as file:
        contents = csv.reader(file, delimiter=',')
        metadata = next(contents)
    device = float(metadata[0])
    alpha = float(metadata[1])
    beta = float(metadata[2])
    drift = float(metadata[3])
    diffusion = float(metadata[4])
    
    return MetaSeries(data,device=device, alpha=alpha, beta=beta, drift=drift, diffusion=diffusion)