import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

def measurement(Bz,t):
    """
    returns either +1 or -1 with probability determined by the model for qubit evolution
    and projective measurement
    
    Parameters:
        Bz: Initial value of DeltaBz in the model (in MHz)
        t: evolution time between state preparation and measurement (in nanoseconds)
    """
    alpha=0.25
    beta=0.67
    BzHertz = Bz*10**6
    tseconds = t*10**(-9)
    pPlus = 1/2*(1+(alpha+beta*np.cos(2*pi*BzHertz*tseconds)))
    x = np.random.rand(1)[0]
    if (pPlus > x):
        x = 1
    else:
        x = -1
    return x
    
def measurementVoltage(Bz,t):
    """
    returns a voltage value from a Gaussian peak based on the qubit state measurement.
    The qubit state measurement is determined by the model for qubit evolution
    and projective measurement.
    
    Parameters:
        Bz: Initial value of DeltaBz in the model (in MHz)
        t: evolution time between state preparation and measurement (in nanoseconds)
    """
    alpha=0.25
    beta=0.67
    Smean = 4.5
    Sstd = 0.3
    Tmean = 5.5
    Tstd = 0.3
    
    BzHertz = Bz*10**6
    tseconds = t*10**(-9)
    pPlus = 1/2*(1+(alpha+beta*np.cos(2*pi*BzHertz*tseconds)))
    x = np.random.rand(1)[0]
    if (pPlus > x):
        voltage = np.random.norm(Smean,Sstd)
    else:
        voltage = np.random.norm(Tmean,Tstd)
    return voltage
    
def measurementDrift(Bz,t,drift,k):
    """
    returns either +1 or -1 with probability determined by the model for qubit evolution
    and projective measurement
    
    Parameters:
        Bz: Initial value of DeltaBz in the model (in MHz)
        t: evolution time between state preparation and measurement (in nanoseconds)
        drift: rate of drift of DeltaBz in kHz/us
        k: measurement index
    """
    alpha=0.25
    beta=0.67
    driftHzs = drift*10**9 #units of Hz/s
    timePerMeasurement = 4*10**(-6)
    
    BzHertz = Bz*10**6
    tseconds = t*10**(-9)
    
    BzCurrent = BzHertz - driftHzs*timePerMeasurement*(k-1)
    pPlus = 1/2*(1+(alpha+beta*np.cos(2*pi*BzCurrent*tseconds)))
    x = np.random.rand(1)[0]
    if (pPlus > x):
        x = 1
    else:
        x = -1
    return x
    
def randomWalk(Bz):
    """
    returns a vector with Bz values that have diffused over time. Each step is 1 microsecond
    so in practice, each consequtive measurement pulls its DeltaBz from every fourth
    value in the vector
    
    Parameters:
        Bz: Initial value of DeltaBz in the model (in MHz)
    """
    D = 7*10**(-6) # need to change from kHz^2 to MHz^2
    
    steps = 10000 # This allows for plenty of time to make ~100 measurements
    BzDiffused = np.empty(steps)
    BzDiffused[0] = Bz
    sigma = np.sqrt(2*D)
    for t in range(1,steps):
        BzDiffused[t] = np.random.normal(BzDiffused[t-1], sigma)
    return BzDiffused
    
def measurementDriftDiffusion(BzDiffused,t,drift,k):
    """
    returns either +1 or -1 with probability determined by the model for qubit evolution
    and projective measurement
    
    Parameters:
        BzDiffused: A vector of Bz values at 1 microsecond increments(in MHz)
        t: evolution time between state preparation and measurement (in nanoseconds)
        drift: rate of drift of DeltaBz in kHz/us
        k: measurement index
    """
    alpha=0.25
    beta=0.67
    driftHzs = drift*10**9 #units of Hz/s
    timePerMeasurement = 4*10**(-6)
    
    
    BzHertz = BzDiffused[4*k]*10**6 # each measurement looks at every 4th value of BzDiffused
                                    # since a measurement takes 4 microseconds
    tseconds = t*10**(-9)
    
    BzCurrent = BzHertz - driftHzs*timePerMeasurement*(k-1)
    pPlus = 1/2*(1+(alpha+beta*np.cos(2*pi*BzCurrent*tseconds)))
    x = np.random.rand(1)[0]
    if (pPlus > x):
        x = 1
    else:
        x = -1
    return x

def likelihood(MetaSeries,Bz):
    alpha = MetaSeries.alpha
    beta = MetaSeries.beta
    drift = MetaSeries.drift*10**9
    diffusion = MetaSeries.diffusion*10**(-6)
    BzHertz = Bz*10**6
    
    timePerMeasurement = 4*10**(-6)
    
    likelihood=1
    
    for i in range(len(MetaSeries)):
        k = MetaSeries[i,0]+1
        tk = MetaSeries[i,1]*10**(-9)
        mk = MetaSeries[i,2]
        
        BzCurrent = BzHertz - drift*timePerMeasurement*(k+1)
        sigma = np.sqrt(2*diffusion*4*k)
        
        likelihoodmk = 1/2*beta*mk*sigma*np.exp(-(2*pi*tk)**2*sigma**2/2)*np.cos(2*pi*BzCurrent*tk)+(1+mk*alpha)/2
        likelihood = likelihood*likelihoodmk
    return likelihood

def getCoefs(MetaSeries):
    """
    returns Fourier coefficients of the likelihood function from a series of measurements.
    This is a discrete Fourier series, but exactly equals the actual likelihood. Essentially,
    it turns a product of cosines into a sum of cosines and sines.
    
    Parameters:
        MetaSeries
    """
    
    alpha = MetaSeries.alpha
    beta = MetaSeries.beta
    drift = MetaSeries.drift*10**9
    diffusion = MetaSeries.diffusion*10**12
    minMeasureTime = 10*10**(-9)
    timePerMeasurement = 4*10**(-6)
    
    c = np.zeros((MetaSeries.shape[0],20001), dtype=complex)
    c[0,10000]=1 # This is like setting a uniform prior

    m = MetaSeries[:,2]
    mtimes = MetaSeries[:,1]
    for k in range(1,MetaSeries.shape[0]):
        n = mtimes[k]
        for i in range(c.shape[1]):
            if (((i+n)<c.shape[1]) and ((i-n)>0)):
                c[k,i] = np.exp(-diffusion*((i-10000)*minMeasureTime)**2*timePerMeasurement+drift*(i-10000)*minMeasureTime*timePerMeasurement*1j)*\
                ((1+m[k]*alpha)*c[k-1,i]+1/2*m[k]*beta*(c[k-1,i+n]+c[k-1,i-n]))
            elif(((i+n)<c.shape[1])):
                c[k,i] = np.exp(-diffusion*((i-10000)*minMeasureTime)**2*timePerMeasurement+drift*(i-10000)*minMeasureTime*timePerMeasurement*1j)*\
                ((1+m[k]*alpha)*c[k-1,i]+1/2*m[k]*beta*(c[k-1,i+n]))  
            else:
                c[k,i] = np.exp(-diffusion*((i-10000)*minMeasureTime)**2*timePerMeasurement+drift*(i-10000)*minMeasureTime*timePerMeasurement*1j)*\
                ((1+m[k]*alpha)*c[k-1,i]+1/2*m[k]*beta*c[k-1,i-n])
    return c

def posterior(coefs,Bzmin,Bzmax,points):
    """
    returns the posterior for the range of DeltaBz values
    
    Parameters:
        coefs
        Bzmin
        Bzmax
        points
    """
    sums=np.ones(points)*np.real(coefs[-1,10000])/2
    Bz = np.linspace(Bzmin,Bzmax,points)
    for j in range(len(Bz)):
        for i in range(0,coefs.shape[1]):
            sums[j] = sums[j] + (np.real(coefs[-1,i])*np.cos(2*pi*Bz[j]*10**6*(i-10000)*10*10**(-9))) - np.imag(coefs[-1,i])*np.sin(2*pi*Bz[j]*10**6*(i-10000)*10*10**(-9))
    plt.plot(Bz,sums)