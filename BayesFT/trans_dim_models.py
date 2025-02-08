'''Class structure for trans-dimensional models used in RJMCMC. Model does not include
hierarchical power law model. Class contains prior, likelihood, and posterior methods. 
Also attributes like parameter domains, labels, etc.'''


from jax import jit
from jax.lax import cond
import jax.numpy as jnp

from BayesFT import functions


class Transdimensional_Model:

    def __init__(self, time_samples, data, max_Nf=None):
        
        # time samples and data
        self.time_samples = time_samples
        self.data = data

        # time sample attributes
        self.Tspan = self.time_samples[-1] - self.time_samples[0]
        self.Nt = self.time_samples.shape[0]

        # maximum number of frequency bins to model
        self.max_Nf = max_Nf
        if self.max_Nf is None:
            self.max_Nf = self.Nt // 2
        self.max_Na = 2 * self.max_Nf

        # frequency attributes of models
        self.models_Nf = jnp.arange(1, self.max_Nf + 1)
        self.models_Na = 2 * self.models_Nf
        self.num_models = self.models_Nf.shape[0]
        self.models_f = [jnp.arange(1, 1 + Nf) / self.Tspan for Nf in self.models_Nf]
        self.models_F = [functions.fast_Fourier_design(self.time_samples, f) for f in self.models_f]

        # maximum likelihood parameteres for each model
        self.models_a_ML = [jnp.linalg.pinv(F) @ self.data for F in self.models_F]
        self.models_ML_time_domain = [F @ a for F, a in zip(self.models_F, self.models_a_ML)]
        self.models_log_tolerance_ML = [jnp.log(jnp.sqrt(jnp.sum((self.data - ML_time_domain)**2.) / self.Nt))
                                        for ML_time_domain in self.models_ML_time_domain]
        self.models_x_ML = [jnp.concatenate((a, jnp.array([log_tol])))
                            for a, log_tol in zip(self.models_a_ML, self.models_log_tolerance_ML)]
        
        # domain of parameters
        self.a_min = -10.
        self.a_max = 10.
        self.log_tol_min = -5.
        self.log_tol_max = 5.
        self.a_range = self.a_max - self.a_min
        self.log_tol_range = self.log_tol_max - self.log_tol_min

        # extremum parameter values
        self.model_x_mins = [jnp.concatenate((jnp.array([self.a_min] * int(Na)), jnp.array([self.log_tol_min])))
                             for Na in self.models_Na]
        self.model_x_maxs = [jnp.concatenate((jnp.array([self.a_max] * int(Na)), jnp.array([self.log_tol_max])))
                             for Na in self.models_Na]
        

        # (log-) parameter volumes to use in uniform prior
        self.log_param_vols = jnp.array([Na * jnp.log(self.a_range) + jnp.log(self.log_tol_range)
                                         for Na in self.models_Na])

        # likelihood and posterior function
        self.fast_lnprior = jit(self.ln_prior)
        self.fast_lnlike = jit(self.ln_likelihood)
        self.fast_lnpost = jit(self.ln_posterior)
        self.fast_lnlike_prior_recovery = jit(self.lnlike_prior_recovery)
        self.fast_lnposterior_prior_recovery = jit(self.lnposterior_prior_recovery)


    # log-prior function model with any Nf (shape of inputs determines model)
    def ln_prior(self, x):
        # check if parameters are in allowed domain
        a = x[:-1]
        log_tolerance = x[-1]
        out_of_bounds = jnp.logical_or(jnp.logical_or(jnp.any(a < self.a_min),
                                                      jnp.any(a > self.a_max)),
                                       jnp.logical_or(log_tolerance < self.log_tol_min,
                                                      log_tolerance > self.log_tol_max))
        # uniform prior
        def in_bounds_case():
            return -self.log_param_vols[x.shape[0] // 2 - 1]
        def out_of_bounds_case():
            return -jnp.inf
        return cond(out_of_bounds, out_of_bounds_case, in_bounds_case)


    # log-likelihood function for any model (shape of any inputs determines model)
    def ln_likelihood(self, x, temperature=1.0):
        # unpack parameters
        a = x[:-1]
        log_tolerance = x[-1]
        tolerance = jnp.exp(log_tolerance)
        # signal realization for parameters
        F = self.models_F[x.shape[0] // 2 - 1]
        signal = F @ a
        lnlike = -0.5 * jnp.sum((self.data - signal)**2.) / tolerance**2. - self.Nt * jnp.log(tolerance)
        return lnlike / temperature
    

    # log-posterior function
    def ln_posterior(self, x, temperature=1.0):
        return self.fast_lnprior(x) + self.fast_lnlike(x, temperature=temperature)
    

    # "likelihood" for prior recovery
    def lnlike_prior_recovery(self, x, temperature=1.0):
        return 1. / temperature


    # "posterior" for prior recovery
    def lnposterior_prior_recovery(self, x, temperature=1.0):
        return self.fast_lnlike_prior_recovery(x, temperature=temperature) + self.fast_lnprior(x)



