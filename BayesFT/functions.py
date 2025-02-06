'''Store commonly used functions.'''


from jax import jit
import jax.numpy as jnp



# Fourier design matrix, input time samples and frqeuency bins
def get_Fourier_design_matrix(t, f):
    # compute arguments for sines and cosines
    argument = 2. * jnp.pi * jnp.outer(t, f)
    # stack into F matrix
    return jnp.concatenate([jnp.sin(argument), jnp.cos(argument)], axis=1)

fast_Fourier_design = jit(get_Fourier_design_matrix)


# diagonal of covariance matrix for power law on Fourier coefficients
def diag_power_law_cov(log_amp, gamma, f):
    amp = jnp.exp(log_amp)
    return amp / jnp.repeat(f, 2)**gamma

fast_diag_power_law_cov = jit(diag_power_law_cov)

