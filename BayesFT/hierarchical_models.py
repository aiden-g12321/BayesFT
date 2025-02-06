'''Class structure for hierarchical models using power law. Class contains prior, likelihood,
and posterior methods. Also attributes like parameter domains, labels, etc.'''


from jax import jit
from jax.lax import cond
import jax.numpy as jnp
import numpy as np
from BayesFT import functions as f



class Hierarchical_Model:

    def __init__(self, time_samples, data, Nf):

        self.time_samples = jnp.array(time_samples)  # discrete time samples as array
        self.data = jnp.array(data)  # data collected over time samples
        self.Nf = Nf  # number of frequency bins to include in model

        # attributes of time samples
        self.Nt = self.time_samples.shape[0]
        self.Tspan = self.time_samples[-1] - self.time_samples[0]
        
        # frequency attributes of model
        self.Na = 2 * self.Nf
        self.f = jnp.arange(1, self.Nf + 1) / self.Tspan
        self.F = f.fast_Fourier_design(self.time_samples, self.f)

        # domain of parameters
        self.log_amp_min = -10.0
        self.log_amp_max = 10.0
        self.gamma_min = 0.0
        self.gamma_max = 10.0
        self.a_min = -10.0
        self.a_max = 10.0
        self.log_sigma_min = -5.0
        self.log_sigma_max = 5.0

        # vectorize extrema
        self.x_min = jnp.concatenate((jnp.array([self.log_amp_min, self.gamma_min]),
                                      jnp.array([self.a_min] * self.Na),
                                      jnp.array([self.log_sigma_min])))
        self.x_max = jnp.concatenate((jnp.array([self.log_amp_max, self.gamma_max]),
                                      jnp.array([self.a_max] * self.Na),
                                      jnp.array([self.log_sigma_max])))
        self.ndim = self.x_min.shape[0]
        
        # logarithm of parameter volume
        self.log_param_vol = jnp.sum(jnp.log(self.x_max - self.x_min))

        # parameter labels for plotting
        self.hyper_labels = np.array([r'$\log A$', r'$\gamma$'])
        self.a_labels = np.array([rf'$a_{{{(i + 2) // 2}}}$' if i % 2 == 0 else rf'$b_{{{(i + 2) // 2}}}$'
                                  for i in range(self.Na)])
        self.log_sigma_label = np.array([r'$\log\sigma$'])
        self.labels = np.concatenate((self.hyper_labels, self.a_labels, self.log_sigma_label))

        # priors, likelihood, and posterior
        self.fast_uniform_lnprior = jit(self.uniform_lnprior)
        self.fast_lnprior_a = jit(self.lnprior_a)
        self.fast_lnprior = jit(self.lnprior)
        self.fast_lnlike = jit(self.lnlikelihood)
        self.fast_lnpost = jit(self.lnposterior)
        self.fast_lnprior_recovery = jit(self.lnprior_recovery)


    # uniform prior
    def uniform_lnprior(self, x):
        out_of_bounds = jnp.logical_or(jnp.any(x < self.x_min),
                                       jnp.any(x > self.x_max))
        def out_of_bounds_case():
            return -jnp.inf
        def in_bounds_case():
            return 0.0
        return cond(out_of_bounds, out_of_bounds_case, in_bounds_case)
    

    # prior on Fourier coefficients
    def lnprior_a(self, x):
        # unpack parameters
        log_amp, gamma = x[:2]
        a = x[2:-1]
        # diagonal of covariance matrix
        diag_of_cov = f.fast_diag_power_law_cov(log_amp, gamma, self.f)
        log_det_cov = jnp.sum(jnp.log(diag_of_cov))
        return -0.5 * jnp.sum(a**2. / diag_of_cov) - 0.5 * log_det_cov


    # combined uniform prior and power law
    def lnprior(self, x):
        return self.fast_uniform_lnprior(x) + self.fast_lnprior_a(x)


    # likelihood
    def lnlikelihood(self, x, temperature=1.0):
        # unpack parameters
        a = x[2:-1]
        log_sigma = x[-1]
        sigma = jnp.exp(log_sigma)
        residual_power = jnp.sum((self.data - self.F @ a)**2.)
        lnlike = -0.5 * residual_power / sigma**2. - self.Nt * log_sigma
        return lnlike / temperature


    # posterior
    def lnposterior(self, x, temperature=1.0):
        return self.fast_lnprior(x) + self.fast_lnlike(x, temperature=temperature)
    
    
    # "posterior" for prior recovery
    def lnprior_recovery(self, x, temperature=1.0):
        return 1. / temperature + self.fast_uniform_lnprior(x)


