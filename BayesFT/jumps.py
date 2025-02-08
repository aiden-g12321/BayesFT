'''Jump proposals for MCMC.'''


import numpy as np
from jax import jit, hessian, vmap
import jax.numpy as jnp
import jax.random as jr
from scipy.stats import multivariate_normal




# Fisher jumps
class FisherJumps:

    def __init__(self, x0, lnpost_func):
        self.x0 = x0  # initial state where to compute Fisher
        self.lnpost_func = lnpost_func  # posterior density

        # store initial Fisher
        self.fast_get_Fisher = jit(self.get_Fisher)
        self.Fisher = self.fast_get_Fisher(self.x0)
        self.vals, self.vecs = jnp.linalg.eigh(self.Fisher)

        # vectorize Fisher jump across chains
        self.fast_Fisher_jump = jit(self.Fisher_jump)
        self.vectorized_Fisher_jump = jit(vmap(self.fast_Fisher_jump, in_axes=(0, None, 0, 0)))

    # compute Fisher numerically
    def get_Fisher(self, x):
        hess = -hessian(self.lnpost_func)(x)
        return hess
    
    # jump along eigenvectors of Fisher
    def Fisher_jump(self, state, iteration, temperature, key):
        keys = jr.split(key, 2)
        # get jump
        direction = jr.choice(keys[0], state.shape[0])
        jump = 1. / jnp.sqrt(jnp.abs(self.vals[direction])) * self.vecs[:, direction]
        jump *= jr.normal(keys[1]) * jnp.sqrt(temperature)
        return state + jump
    
    

# Differential evolution
class DifferentialEvolution:

    def __init__(self, len_history, x_min, x_max):
        self.len_history = len_history  # how many samples in adaptive history
        self.x_min = x_min
        self.x_max = x_max
        self.ndim = self.x_min.shape[0]
        self.jump_weight = 2.38 / jnp.sqrt(2. * self.ndim)

        # initialize adaptive history
        self.history = jr.uniform(jr.PRNGKey(22), minval=self.x_min, maxval=self.x_max,
                                  shape=(self.len_history, self.ndim))
        
        self.fast_DE_jump = jit(self.DE_jump)

        # vectorize jump over chains
        self.vectorized_DE_jump = jit(vmap(self.fast_DE_jump, in_axes=(0, None, 0, 0)))
        
    def DE_jump(self, state, iteration, temperature, key):
        # split random keys
        DE_keys = jr.split(key, 5)
        # get jump
        draw1 = jr.choice(DE_keys[0], self.history)
        draw2 = jr.choice(DE_keys[1], self.history)
        jump = jr.normal(DE_keys[2]) * self.jump_weight * (draw1 - draw2) * 0.1
        jump += jr.normal(DE_keys[3], shape=(self.ndim,)) * 0.00001
        # move to new state
        new_state = state + jump
        # update history
        self.history = self.history.at[jr.choice(DE_keys[4], self.len_history)].set(jnp.copy(state))
        return new_state


def zero_jump(state, iteration, temp, key):
    return state

# vectorize zero jump
vectorized_zero_jump = vmap(zero_jump, in_axes=(0, None, 0, 0))


#############################################################################################################
#################################### jumps for trans-dimensional MCMC #######################################
#############################################################################################################
# (note some jumps are still within model and identical to those above,
# but have different inputs / outputs ammenable to RJMCMC methods)

# Fisher jumps
class Trans_dim_jumps:

    def __init__(self, xs, lnlike_func, lnpost_func, max_model_ndx, a_min, a_max):
        self.xs = xs  # states where to compute Fisher for each model
        self.num_models = len(self.xs)
        self.lnlike_func = lnlike_func
        self.lnpost_func = lnpost_func
        self.max_model_ndx = max_model_ndx
        self.a_min = a_min
        self.a_max = a_max

        # Fishers for models
        self.fast_get_Fisher = jit(self.get_Fisher)
        self.models_Fisher = [self.fast_get_Fisher(x) for x in self.xs]

        # multivariate Gaussians with covariance estimated by Fisher
        self.models_Gauss = [multivariate_normal(x, np.linalg.inv(Fisher))
                             for x, Fisher in zip(self.xs, self.models_Fisher)]

        # jumps along eigenvectors of Fisher
        self.models_jumps = []
        for Fisher in self.models_Fisher:
            vals, vecs = jnp.linalg.eigh(Fisher)
            jumps = np.zeros(Fisher.shape)
            for i in range(vals.shape[0]):
                jumps[i] = 1. / jnp.sqrt(jnp.abs(vals[i])) * vecs[:, i]
            self.models_jumps.append(jnp.array(jumps))

    # compute Fisher numerically
    def get_Fisher(self, x):
        hess = -hessian(self.lnpost_func)(x)
        return hess
    
    # jump along eigenvectors of Fisher
    def Fisher_jump(self, chain, key):
        keys = jr.split(key, 2)
        state, lnpost = chain.state, chain.lnpost
        # get jump
        direction = jr.choice(keys[0], state.shape[0])
        jump = self.models_jumps[state.shape[0] // 2 - 1][direction]
        jump *= jr.normal(keys[1]) * jnp.sqrt(chain.temperature)
        new_state = state + jump
        new_lnpost = self.lnpost_func(new_state, temperature=chain.temperature)
        accept_prob = jnp.exp(new_lnpost - lnpost)
        return new_state, new_lnpost, accept_prob
    
    
    def uniform_trans_dim_jump(self, chain, key):
        keys = jr.split(key, 2)
        model_ndx = chain.state.shape[0] // 2 - 1
        if (jr.choice(keys[0], 2) == 0 or model_ndx == 0) and model_ndx != self.max_model_ndx:  # add Fourier mode
            new_coeff = jr.uniform(keys[1], shape=(2,), minval=self.a_min, maxval=self.a_max)
            new_state = jnp.concatenate((chain.state[:-1], new_coeff, jnp.array([chain.state[-1]])))
        else:  # remove Fourier mode
            new_state = jnp.concatenate((chain.state[:-3], jnp.array([chain.state[-1]])))
        new_lnpost = self.lnpost_func(new_state, temperature=chain.temperature)
        acc_prob = np.exp(self.lnlike_func(new_state) - self.lnlike_func(chain.state))
        if model_ndx == 0 or model_ndx == self.max_model_ndx:
            acc_prob /= 2.
        return new_state, new_lnpost, acc_prob
    

    def ML_trans_dim_jump(self, chain, key):
        current_model_ndx = chain.state.shape[0] // 2 - 1
        new_model_ndx = jr.choice(key, self.max_model_ndx)
        new_state = self.models_Gauss[new_model_ndx].rvs()
        new_lnpost = self.lnpost_func(new_state, temperature=chain.temperature)
        log_acc_prob = new_lnpost - chain.lnpost
        log_acc_prob += self.models_Gauss[current_model_ndx].logpdf(chain.state)
        log_acc_prob -= self.models_Gauss[new_model_ndx].logpdf(new_state)
        acc_prob = np.exp(log_acc_prob)
        return new_state, new_lnpost, acc_prob
    

