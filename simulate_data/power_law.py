'''Script to simulate time-domain data obeying a power law.'''


import numpy as np
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from BayesFT import functions
from simulate_data.data_structure import Data

# default time samples over which data is collected
# (100 samples from 0 sec to 1 sec)
default_time_samples = jnp.linspace(0., 1., 100)


# simulate time-domain data obeying power law.
def sim_power_law(log_amp,  # log-amplitude of power law
                  gamma,  # spectral index of power law
                  times=default_time_samples,  # time samples
                  noise_log_stdev=-100.0,  # log-standard-deviation of white noise added
                  random_seed=3,  # random seed
                  plot=True  # plot simulated data in time- and frequency-domain
                  ):

    # make random keys for coefficient and noise generation
    coeff_key, noise_key = jr.split(jr.PRNGKey(random_seed))

    # attributes of time samples
    Nt = times.shape[0]
    Tspan = times[-1] - times[0]

    # frequency bins (neglecting zero-frequency)
    Nf = Nt // 2
    Na = 2 * Nf
    freqs = jnp.arange(1, Nf + 1) / Tspan
    logf = jnp.log(freqs)

    # diagonal of covariance matrix of Fourier coefficients
    diag_cov_inj = functions.fast_diag_power_law_cov(log_amp, gamma, freqs)

    # use Cholesky decomposition to make correlated Fourier coefficients
    a = jnp.sqrt(diag_cov_inj) * jr.normal(coeff_key, shape=(Na,))

    # check accuracy of injection by averaging over many realizations
    many_a = jnp.sqrt(diag_cov_inj) * jr.normal(coeff_key, shape=(10_000, Na))

    # simulate power law signal in time-domain
    F = functions.fast_Fourier_design(times, freqs)
    signal_inj = F @ a

    # linearly add zero-mean Gaussian noise to make data
    noise_stdev = jnp.exp(noise_log_stdev)
    noise = jr.normal(noise_key, shape=(Nt,)) * noise_stdev
    data = signal_inj + noise
    
    # compute Fourier coefficients which maximize likelihood to estimate white noise
    noise_a_ML = jnp.linalg.pinv(F) @ noise

    # compute PSDs of various processes for plotting
    power_law_inj_logPSD = jnp.log(diag_cov_inj[::2] + diag_cov_inj[1::2])
    coeff_inj_logPSD = jnp.log(a[::2]**2. + a[1::2]**2.)
    average_realizations_logPSD = jnp.log(jnp.mean(many_a[:,::2]**2. + many_a[:,1::2]**2., axis=0))
    noise_logPSD = jnp.log(noise_a_ML[::2]**2. + noise_a_ML[1::2]**2.)
    
    # plot data
    if plot:

        # time-domain
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(times, data, color='C0', s=5, label='data')
        plt.plot(times, signal_inj, color='C1', label='injection')
        plt.xlabel('time [s]')
        plt.ylabel('signal [units]')
        plt.legend()

        # frequency-domain
        plt.subplot(1, 2, 2)
        plt.plot(logf, power_law_inj_logPSD, color='green', label='power law')
        plt.scatter(logf, coeff_inj_logPSD, color='C1', s=10, label='injection')
        plt.scatter(logf, average_realizations_logPSD, color='red', s=10, 
                    label='average of many realizations')
        plt.plot(logf, noise_logPSD, color='k', label='white noise estimate')
        plt.xlabel('log(frequency)')
        plt.ylabel('log(PSD)')
        plt.legend()
        plt.show()
    
    return Data(times, data)

