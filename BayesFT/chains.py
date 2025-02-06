'''Class structure for MCMC chains. Used only for trans-dimensional MCMC
where samples may be different sizes. For fixed dimension MCMC usual numpy.arrays
are used to store chain samples.'''


import numpy as np


class Chain:


    def __init__(self, x0, lnpost0, temperature):
        # initial point in parameter space and log-posterior value
        self.x0 = x0
        self.lnpost0 = lnpost0

        # temperature of chain
        self.temperature = temperature
        self.sqrt_temperature = np.sqrt(temperature)
    
        # initialize parameter samples and posterior values
        self.samples = [self.x0]
        self.lnposts = [self.lnpost0]

        # current state of chain
        self.state = self.samples[-1]
        self.lnpost = self.lnposts[-1]
    

    # add sample to chain
    def add_sample(self, sample, lnpost):
        self.state = sample
        self.samples.append(self.state)
        self.lnpost = lnpost
        self.lnposts.append(self.lnpost)

