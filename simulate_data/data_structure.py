'''Class structure for simulated data.'''


import numpy as np
from BayesFT import functions


class Data:

    def __init__(self, time_samples, time_domain_data):
        self.time_samples = np.array(time_samples)  # time samples over which data is collected
        self.data_values = np.array(time_domain_data)  # data in time-domain

        # attributes of time samples
        self.t0 = self.time_samples[0]
        self.t1 = self.time_samples[-1]
        self.Nt = self.time_samples.shape[0]
        self.Tspan = self.t1 - self.t0
        self.dt = self.time_samples[1] - self.t0

        # attributes of frequency bins (neglecting zero-frequency)
        self.Nf = self.Nt // 2
        self.freq_bins = np.arange(1, self.Nf + 1) / self.Tspan
        self.F_matrix = functions.fast_Fourier_design(self.time_samples, self.freq_bins)

        # maximum likelihood Fourier coefficients
        self.a_ML = np.linalg.pinv(self.F_matrix) @ self.data_values
        self.logPSD_ML = np.log(self.a_ML[::2]**2. + self.a_ML[1::2]**2.)

