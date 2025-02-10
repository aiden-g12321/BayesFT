'''Class structure for hierarchical trans-dimensional models used in RJMCMC. Class contains prior, 
likelihood, and posterior methods. Also attributes like parameter domains, labels, etc.'''


from jax import jit
from jax.lax import cond
import jax.numpy as jnp

from BayesFT import functions


class Hierarch_trans_model:

    def __init__(self, time_samples, data, hypers_inj, max_Nf=None):
        
        # time samples and data
        self.time_samples = time_samples
        self.data = data

        self.hypers_inj = hypers_inj

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
        self.model_ndxs = jnp.arange(self.num_models)
        self.max_model_ndx = jnp.max(self.model_ndxs)
        self.models_f = [jnp.arange(1, 1 + Nf) / self.Tspan for Nf in self.models_Nf]
        self.models_F = [functions.fast_Fourier_design(self.time_samples, f) for f in self.models_f]

        # maximum likelihood parameteres for each model
        self.models_hyper_ML = self.hypers_inj
        self.models_a_ML = [jnp.linalg.pinv(F) @ self.data for F in self.models_F]
        self.models_ML_time_domain = [F @ a for F, a in zip(self.models_F, self.models_a_ML)]
        self.models_log_tolerance_ML = [jnp.log(jnp.sqrt(jnp.sum((self.data - ML_time_domain)**2.) / self.Nt))
                                        for ML_time_domain in self.models_ML_time_domain]
        self.models_x_ML = [jnp.concatenate((self.models_hyper_ML, jnp.array([log_tol]), a))
                            for log_tol, a in zip(self.models_log_tolerance_ML, self.models_a_ML)]
        
        # domain of parameters
        self.log_amp_min = -5.0
        self.log_amp_max = 15.0
        self.gamma_min = 0.0
        self.gamma_max = 15.0
        self.log_tol_min = -5.
        self.log_tol_max = 5.
        self.a_min = -10.
        self.a_max = 10.

        # extremum parameter values
        self.model_x_mins = [jnp.concatenate((jnp.array([self.log_amp_min, self.gamma_min]),
                                              jnp.array([self.log_tol_min]),
                                              jnp.array([self.a_min] * int(Na))))
                             for Na in self.models_Na]
        self.model_x_maxs = [jnp.concatenate((jnp.array([self.log_amp_max, self.gamma_max]),
                                              jnp.array([self.log_tol_max]),
                                              jnp.array([self.a_max] * int(Na))))
                             for Na in self.models_Na]
        

        # (log-) parameter volumes to use in uniform prior
        self.log_param_vols = jnp.array([jnp.sum(jnp.log(x_max - x_min))
                                         for x_min, x_max in zip(self.model_x_mins, self.model_x_maxs)])

        # likelihood and posterior function
        self.fast_uniform_lnprior = jit(self.uniform_lnprior)
        self.fast_lnprior_a = jit(self.lnprior_a)
        self.fast_lnprior = jit(self.ln_prior)
        self.fast_lnlike = jit(self.ln_likelihood)
        self.fast_lnpost = jit(self.ln_posterior)
        self.fast_lnlike_prior_recovery = jit(self.lnlike_prior_recovery)
        self.fast_lnposterior_prior_recovery = jit(self.lnposterior_prior_recovery)

    # log-prior function model with any Nf (shape of inputs determi˜nes model)
    def uniform_lnprior(self, x):
        # check if parameters are in allowed domain
        log_amp, gamma = x[:2]
        log_tolerance = x[2]
        a = x[3:]
        out_of_bounds = jnp.logical_or(jnp.logical_or(jnp.logical_or(log_amp < self.log_amp_min,
                                                                     log_amp > self.log_amp_max),
                                                      jnp.logical_or(gamma < self.gamma_min,
                                                                     gamma > self.gamma_max)),
                                       jnp.logical_or(jnp.logical_or(jnp.any(a < self.a_min),
                                                                     jnp.any(a > self.a_max)),
                                                      jnp.logical_or(log_tolerance < self.log_tol_min,
                                                                     log_tolerance > self.log_tol_max)))
        # uniform prior
        def in_bounds_case():
            return -self.log_param_vols[a.shape[0] // 2 - 1]
        def out_of_bounds_case():
            return -jnp.inf
        return cond(out_of_bounds, out_of_bounds_case, in_bounds_case)
    

    # prior on Fourier coefficients
    def lnprior_a(self, x):
        # unpack parameters
        log_amp, gamma = x[:2]
        a = x[2:-1]
        # diagonal of covariance matrix
        model_ndx = a.shape[0] // 2 - 1
        diag_of_cov = functions.fast_diag_power_law_cov(log_amp, gamma, self.models_f[model_ndx])
        log_det_cov = jnp.sum(jnp.log(diag_of_cov))
        return -0.5 * jnp.sum(a**2. / diag_of_cov) - 0.5 * log_det_cov
    

    # combined prior including uniform contribution and hierarchical model
    def ln_prior(self, x):
        return self.fast_uniform_lnprior(x) + self.fast_lnprior_a(x)
    

    # log-likelihood function for any model (shape of any inputs determines model)
    def ln_likelihood(self, x, temperature=1.0):
        # unpack parameters
        log_tolerance = x[2]
        tolerance = jnp.exp(log_tolerance)
        a = x[3:]
        # signal realization for parameters
        F = self.models_F[a.shape[0] // 2 - 1]
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



