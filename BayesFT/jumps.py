from jax import jit, hessian, vmap
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.linalg import block_diag


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

