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
    
def randomWalk(Bz,diffusion):
    """
    returns a vector with Bz values that have diffused over time. Each step is 1 microsecond
    so in practice, each consequtive measurement pulls its DeltaBz from every fourth
    value in the vector
    
    Parameters:
        Bz: Initial value of DeltaBz in the model (in MHz)
    """
    D = diffusion*10**(-6) # need to change from kHz^2 to MHz^2
    
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
    """
    returns a likelihood using the conventional product of individual measurements
    This is not so useful anymore, but useful in checking that the Fourier
    method works.
    
    Parameters:
        MetaSeries
        Bz (value from which to start drift and diffusion
    """
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

    ms = MetaSeries[:,2]
    mtimes = MetaSeries[:,1]/10
    diffusionVector = np.array( [np.exp(-diffusion*((i-10000)*2*np.pi*minMeasureTime)**2*timePerMeasurement) for i in range(c.shape[1])])
    driftVector = np.array( [np.exp(drift*(i-10000)*2*np.pi*minMeasureTime*timePerMeasurement*1j) for i in range(c.shape[1])])
           
    for k in range(1,c.shape[0]):
        n = mtimes[k]
        m = ms[k]
        cplus = np.concatenate((c[k-1,1:n+1][::-1],c[k-1,:-n]))
        cminus = np.concatenate((c[k-1,n:],c[k-1,-n-1:-1][::-1]))
        c[k,:] = (1+m*alpha)*c[k-1,:]+1/2*m*beta*(cplus+cminus)
        c[k,:] = np.multiply(c[k,:],diffusionVector,driftVector)
        c[k,:] = c[k,:]/c[k,10000]/50
    return c
    
def simulateCoefs(alpha, beta, drift, diffusion, BzDiffused, sequence=np.arange(1,101)):
    """
    returns the coefficients from a series of measurements with the evolution
    time predetermined
    Parameters:
        alpha
        beta
        drift (in kHz/us)
        diffusion (in kHz^2/us)
        BzDiffused (the actual Bz to be estimated)
        sequence (evolution times for each measurement in 10*ns)
    """
    minMeasureTime = 10*10**(-9)
    timePerMeasurement = 4*10**(-6)
    drift = drift*10**9
    diffusion = diffusion*10**12 
            
    if drift == 0:
        c = np.zeros((len(sequence),10000))
        c[0,0]=1
        
        diffusionVector = np.array( [np.exp(-diffusion*(i*2*np.pi*minMeasureTime)**2*timePerMeasurement) for i in range(c.shape[1])])
        driftVector = np.array( [np.exp(drift*(i-10000)*2*np.pi*minMeasureTime*timePerMeasurement*1j) for i in range(c.shape[1])])
        
        m = np.empty(c.shape[0])
        for k in range(1,c.shape[0]):
            n = sequence[k]
            m[k] = measurementDriftDiffusion(BzDiffused,n*10,drift,k)
            cplus = np.concatenate((c[k-1,1:n+1][::-1],c[k-1,:-n]))
            cminus = np.concatenate((c[k-1,n:],c[k-1,-n-1:-1][::-1]))
            c[k,:] = (1+m[k]*alpha)*c[k-1,:]+1/2*m[k]*beta*(cplus+cminus)
            c[k,:] = np.multiply(c[k,:],diffusionVector)
    else:
        c = np.zeros((len(sequence),20001), dtype=complex)
        c[0,10000]=1 # This is like setting a uniform prior
        
        diffusionVector = np.array( [np.exp(-diffusion*(i*minMeasureTime)**2*timePerMeasurement) for i in range(c.shape[1])])
        driftVector = np.array( [np.exp(drift*(i-10000)*minMeasureTime*timePerMeasurement*1j) for i in range(c.shape[1])])
        
        m = np.empty(c.shape[0])   
        for k in range(1,c.shape[0]):
            n = sequence[k]
            m[k] = measurementDriftDiffusion(BzDiffused,n*10,drift,k)
            cplus = np.concatenate((c[k-1,1:n+1][::-1],c[k-1,:-n]))
            cminus = np.concatenate((c[k-1,n:],c[k-1,-n-1:-1][::-1]))
            c[k,:] = (1+m[k]*alpha)*c[k-1,:]+1/2*m[k]*beta*(cplus+cminus)
            c[k,:] = np.multiply(c[k,:],diffusionVector,driftVector)
        
    return c
    
def optimizedCoefs(length, alpha, beta, drift, diffusion, BzDiffused):
    """
    returns the coefficients from a series of measurements with the evolution
    time optimized before each measurement
    Parameters:
        length (number of measurements)
        alpha
        beta
        drift (in kHz/us)
        diffusion (in kHz^2/us)
        BzDiffused (the actual Bz to be estimated)
    """
    minMeasureTime = 10*10**(-9)
    timePerMeasurement = 4*10**(-6)
    drift = drift*10**9
    diffusion = diffusion*10**12
    
            
    c = np.zeros((length,10000))
    c[0,0]=1
    
    diffusionVector = np.array( [np.exp(-diffusion*(i*2*np.pi*minMeasureTime)**2*timePerMeasurement) for i in range(c.shape[1])])
    driftVector = np.array( [np.exp(drift*(i-10000)*2*np.pi*minMeasureTime*timePerMeasurement*1j) for i in range(c.shape[1])])
    
    m = np.empty(c.shape[0])
    ns = np.empty(c.shape[0])
    for k in range(1,c.shape[0]):
        n = np.argmin([expVar(c[k-1,:],i,diffusionVector,alpha,beta) for i in range(1,100+2*k)])+1
        ns[k] = n
        print(k,':',n)
        m[k] = measurementDriftDiffusion(BzDiffused,n*10,drift,k)
        cplus = np.concatenate((c[k-1,1:n+1][::-1],c[k-1,:-n]))
        cminus = np.concatenate((c[k-1,n:],c[k-1,-n-1:-1][::-1]))
        c[k,:] = (1+m[k]*alpha)*c[k-1,:]+1/2*m[k]*beta*(cplus+cminus)
        c[k,:] = np.multiply(c[k,:],diffusionVector)
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
    return Bz[np.argmax(sums)]

def Bzvariance(coefs):
    """
    returns the variance of DeltaBz using just the Fourier coefficients of the posterior
    
    Parameters:
        coefs
    """
    exp2 = (50**2)/3
    for i in range(1,10000):
        exp2 = exp2 + 2*coefs[i]*(10**2)*50/((np.pi*i)**2)*((-1)**i)/coefs[0]
    exp = 50/2 # expectation value of DeltaBz
    for i in range(1,10000):
        exp = exp + coefs[i]*100/((np.pi*i)**2)*((-1)**i-1)/coefs[0]
    return (exp2-exp**2)

def expectedBz(coefs, allreal=True):
    """
    returns the expectation value for DeltaBz just based on the Fourier coefficients
    Parameters:
        coefs
        allreal: can account for complex coefficients in the case of drift
    """
    if allreal==True:
        exp = (100**2-50**2)/(2*50) # expectation value of DeltaBz
        for i in range(1,10000):
            exp = exp - coefs[i]*100/((np.pi*i)**2)*((-1)**i-1)/coefs[0]
    else:
        exp = (100**2-50**2)/(2*50) # expectation value of DeltaBz
        for i in range(1,10000):
            exp = exp - np.real(coefs[i+10000])*100/((np.pi*i)**2)*((-1)**i-1)/np.real(coefs[10000])
    return exp

def diffusionVector(diffusion, length):
    """
    returns a vector to multiply by a coefficient vector to represent diffusion of the posterior over time
    Parameters:
        diffusion (in kHz^2/uS)
        length (of the coefficient vector)
    """
    D = diffusion*10**12
    minMeasureTime = 10*10**(-9)
    timePerMeasurement = 4*10**(-6)

    return np.array([np.exp(-D*(i*2*np.pi*minMeasureTime)**2*timePerMeasurement) for i in range(1,length+1)])
    
def expVar(inputc,n,diffusionVector,alpha,beta):
    """
    returns the expectation value for the variance in DeltaBz if just based on the Fourier coefficients
    Parameters:
        inputc (Fourier coefficients)
        n (evolution time of the next measurement in 10*ns)
        diffusionVector
        alpha
        beta
    """
    cplus = np.concatenate((inputc[1:n+1][::-1],inputc[:-n]))
    cminus = np.concatenate((inputc[n:],inputc[-n-1:-1][::-1]))
    cS = (1+1*alpha)*inputc+1/2*1*beta*(cplus+cminus)
    cS = np.multiply(cS,diffusionVector)
    cT = (1-1*alpha)*inputc-1/2*1*beta*(cplus+cminus)
    cT = np.multiply(cT,diffusionVector)
    return (cS[0]*Bzvariance(cS)+cT[0]*Bzvariance(cT))/(cS[0]+cT[0])

def getMSE(coefs,BzDiffused):
    """
    returns the mean squared error after each measurement averaged over many experiments
    Parameters:
        coefs
        BzDiffused for comparing to the actual Bz value
    """
    mse = np.empty(100)
    for i in range(coefs.shape[1]):
        expBz = [expectedBz(coefs[j,i,:]) for j in range(coefs.shape[0])]
        mse[i] = np.mean(np.square(np.subtract(expBz,BzDiffused[4*i])))  
    return mse