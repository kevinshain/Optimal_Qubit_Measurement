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

def measurementDrift(Bz,t,k):
    """
    returns either +1 or -1 with probability determined by the model for qubit evolution
    and projective measurement
    
    Parameters:
        Bz: Initial value of DeltaBz in the model (in MHz)
        t: evolution time between state preparation and measurement (in nanoseconds)
        k: measurement index
    """
    alpha=0.25
    beta=0.67
    drift = 2.5*10**8 #units of Hz/s
    timePerMeasurement = 4*10**(-6)
    
    BzHertz = Bz*10**6
    tseconds = t*10**(-9)
    
    BzCurrent = BzHertz + drift*timePerMeasurement*(k-1)
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
    
def measurementDriftDiffusion(BzDiffused,t,k):
    """
    returns either +1 or -1 with probability determined by the model for qubit evolution
    and projective measurement
    
    Parameters:
        BzDiffued: A vector of Bz values at 1 microsecond increments(in MHz)
        t: evolution time between state preparation and measurement (in nanoseconds)
        k: measurement index
    """
    alpha=0.25
    beta=0.67
    drift = 2.5*10**8 #units of Hz/s
    timePerMeasurement = 4*10**(-6)
    
    
    BzHertz = BzDiffused[4*k]*10**6 # each measurement looks at every 4th value of BzDiffused
                                    # since a measurement takes 4 microseconds
    tseconds = t*10**(-9)
    
    BzCurrent = BzHertz + drift*timePerMeasurement*(k-1)
    pPlus = 1/2*(1+(alpha+beta*np.cos(2*pi*BzCurrent*tseconds)))
    x = np.random.rand(1)[0]
    if (pPlus > x):
        x = 1
    else:
        x = -1
    return x