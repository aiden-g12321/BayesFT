# %% [markdown]
# # Demonstrate bias in hierarchical modeling of power law
# 
# ### We construct time-domain data obeying a power law. The data is modeled using a collection of Fourier coefficients, and hierarchically modeled with an amplitude and spectral index which describe the power law. We observe bias in the hyper-parameters depending on the number of frequency bins included in the model.

# %% [markdown]
# # Simulate data

# %%
from jax import vmap
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from corner import corner
import sys
sys.path.append('../')
from BayesFT import functions as f
from BayesFT import hierarchical_models as hm
from BayesFT.PTMCMC import PTMCMC
from BayesFT import jumps
from copy import deepcopy


from eryn.ensemble import EnsembleSampler
from eryn.state import State
from eryn.prior import uniform_dist, ProbDistContainer
import h5py
import corner
from eryn.moves import GaussianMove, StretchMove
# %%
# time series over which data is collected
t0 = 0.
t1 = 1.
Tspan = t1 - t0
Nt = 100
t = jnp.linspace(t0, t1, Nt)

# frequency bins injected into data (neglecting zero frequency)
Nf_inj = t.shape[0] // 2

model_1_Nf = 10  # number of frequency bins in model 1
Na_inj = 2 * Nf_inj # Aiden choice
# Na_inj = model_1_Nf # consistent with model_1_Nf

f_inj = jnp.arange(1, Nf_inj + 1) / Tspan
logf_inj = jnp.log(f_inj)
df_inj = f_inj[1] - f_inj[0]

# %%
# injected hyper-parameters
log_amp_inj = 1.4
gamma_inj = 3.3
hypers_inj = jnp.array([log_amp_inj, gamma_inj])
diag_cov_inj = f.fast_diag_power_law_cov(log_amp_inj, gamma_inj, f_inj)

# use Cholesky decomposition to make correlated Fourier coefficients
a_seed = jr.PRNGKey(4)  # seed for injected Fourier coefficients
np.random.seed(4)
print('diag_cov_inj', diag_cov_inj.shape, "Na_inj", Na_inj)
# a_inj = np.sqrt(diag_cov_inj) * np.random.normal(0.0, size=(Na_inj,)) # np.random.multivariate_normal(np.zeros_like(diag_cov_inj),np.diag(diag_cov_inj))# 
# a_inj[2*model_1_Nf:] = 0.0
# a_inj = jnp.array(a_inj)

# shape of a_inj is (2*Nf_inj,)
# assert a_inj.shape[0] == Na_inj
a_inj = np.sqrt(diag_cov_inj) * jr.normal(a_seed, shape=(Na_inj,))#np.random.multivariate_normal(np.zeros_like(diag_cov_inj),np.diag(diag_cov_inj))# 

# check accuracy of injection by averaging over many realizations
many_a = np.sqrt(diag_cov_inj) * jr.normal(a_seed, shape=(10_000, Na_inj))

# simulate power law signal in time-domain
F_inj = f.fast_Fourier_design(t, f_inj)
signal_inj = F_inj @ a_inj

# add zero-mean Gaussian noise
log_sigma_inj = 0.0
sigma_inj = np.exp(log_sigma_inj)
noise_seed = jr.PRNGKey(10)
noise = jr.normal(noise_seed, shape=(Nt,)) * sigma_inj

# make and plot data
data = signal_inj + noise
# data -= jnp.mean(data)

# get ML coefficients which describe noise
noise_a_ML = jnp.linalg.pinv(F_inj) @ noise

# time-domain
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(t, data, color='C0', s=5, label='data')
plt.plot(t, signal_inj, color='C1', label='injection')
plt.xlabel('time [s]')
plt.ylabel('signal [units]')
plt.legend()

# frequency-domain
plt.subplot(1, 2, 2)
plt.plot(logf_inj, np.log(diag_cov_inj[::2] + diag_cov_inj[1::2]), color='green', label='power law')
plt.scatter(logf_inj, np.log(a_inj[::2]**2. + a_inj[1::2]**2.), color='C1', s=10, label='injection')
plt.scatter(logf_inj, np.log(np.mean(many_a[:,::2]**2. + many_a[:,1::2]**2., axis=0)), color='red', s=10, 
            label='average of many realizations')
plt.plot(logf_inj, np.log(noise_a_ML[::2]**2. + noise_a_ML[1::2]**2.), color='k', label='white noise estimate')
plt.xlabel('log(frequency)')
plt.ylabel('log(PSD)')
plt.legend()
plt.savefig('data.pdf')


# # Model specification
# compare bias in two models
model_1 = hm.Hierarchical_Model(t, data, model_1_Nf)

# point near maximum likelihood (ML) solution
a_ML_1 = jnp.linalg.pinv(model_1.F) @ data
log_sigma_ML_1 = jnp.log(jnp.sqrt(jnp.sum((data - model_1.F @ a_ML_1)**2.) / Nt))
x_ML_1 = jnp.concatenate((hypers_inj, a_ML_1, jnp.array([log_sigma_inj])))

# %% [markdown]
# # MCMC

# %%
# jump proposals: Fisher and differential evolution
Fisher_1 = jumps.FisherJumps(x_ML_1, model_1.fast_lnpost)
DEjumps_1 = jumps.DifferentialEvolution(100, model_1.x_min, model_1.x_max)
jump_proposals_1 = [[Fisher_1.vectorized_Fisher_jump, 20],
                      [DEjumps_1.vectorized_DE_jump, 0]]

from scipy.special import gamma

class InverseGammaDistribution(object):
    """Generate inverse gamma distribution with shape ``alpha`` and scale ``beta``

    Args:
        alpha (double): Shape parameter of the inverse gamma distribution
        beta (double): Scale parameter of the inverse gamma distribution
        use_cupy (bool, optional): If ``True``, use CuPy. If ``False`` use Numpy.
            (default: ``False``)

    Raises:
        ValueError: Issue with inputs.

    """

    def __init__(self, alpha, beta, use_cupy=False):
        if alpha <= 0 or beta <= 0:
            raise ValueError("Alpha and beta must be positive.")

        self.alpha = alpha
        self.beta = beta

        self.use_cupy = use_cupy
        if use_cupy:
            try:
                cp.abs(1.0)
            except NameError:
                raise ValueError("CuPy not found.")

    def rvs(self, size=1):
        if not isinstance(size, int) and not isinstance(size, tuple):
            raise ValueError("size must be an integer or tuple of ints.")

        if isinstance(size, int):
            size = (size,)

        xp = np if not self.use_cupy else cp

        out = 1 / xp.random.gamma(self.alpha, 1 / self.beta, size=size)

        return out

    def pdf(self, x):
        xp = np if not self.use_cupy else cp

        out = (self.beta ** self.alpha) / gamma(self.alpha) * x ** (-self.alpha - 1) * xp.exp(-self.beta / x)
        out[x <= 0] = 0

        return out

    def logpdf(self, x):
        xp = np if not self.use_cupy else cp

        out = xp.zeros_like(x)
        valid = x > 0
        out[valid] = (self.alpha * xp.log(self.beta) - xp.log(gamma(self.alpha)) - (self.alpha + 1) * xp.log(x[valid]) - self.beta / x[valid])
        out[~valid] = -xp.inf

        return out

    def copy(self):
        return deepcopy(self)

# ProbDistContainer
inv = InverseGammaDistribution(1.0, 1.0)
# breakpoint()
priors = {
    "powerlaw": ProbDistContainer({
        0: uniform_dist(model_1.log_amp_min, model_1.log_amp_max),  # amplitude
        1: uniform_dist(model_1.gamma_min, model_1.gamma_max),  # slope
    }),
    "noise": ProbDistContainer({
        # 0: uniform_dist(model_1.log_sigma_min, model_1.log_sigma_max),
        0: inv
    }),
}
# uni_dist_a = uniform_dist(-2.0, 2.0)
uni_dist_a = uniform_dist(model_1.a_min, model_1.a_max)
for n in range(model_1_Nf):
    priors[f"coeff_{n}"] = ProbDistContainer({ii: uni_dist_a for ii in range(2)})


nwalkers = 64
ntemps = 12
ndims = {"powerlaw": 2, "noise": 1}
branch_names = ["powerlaw", "noise"]
nleaves_max = {"powerlaw":  1, "noise": 1}
nleaves_min = {"powerlaw":  1, "noise": 1}
for n in range(model_1_Nf):
    ndims[f"coeff_{n}"] = 2
    branch_names.append(f"coeff_{n}")
    
    nleaves_max[f"coeff_{n}"] = 1
    # nleaves_min[f"coeff_{n}"] = 1
    if n < 2:# we need at least two coefficients to fit an amplitude and slope
        nleaves_min[f"coeff_{n}"] = 1
    else:
        nleaves_min[f"coeff_{n}"] = 0 # 1

zero_array = np.zeros(ndims["powerlaw"]*nleaves_max["powerlaw"] + ndims["noise"]*nleaves_max["noise"] + 2*model_1_Nf)
f_vec = np.array(f_inj)

def loglike(x):
    # log sigma
    # return 0.0
    x_fixed = zero_array.copy()
    x_fixed[:2] = x[0]
    x_fixed[-1] = np.log(x[1])

    # remove the first two branches
    sum_lnprior_a = 0.0
    for n in range(len(x)-2):
        if x[n+2] is not None:
            # print(n, x[n+2])
            x_fixed[2+2*n:4+2*n] = x[n+2]
            diag_of_cov_here = np.exp(x_fixed[0]) / f_vec[n]**x_fixed[1]
            log_det_cov_here = np.log(diag_of_cov_here)
            sum_lnprior_a += -0.5 * np.sum(x_fixed[2+2*n:4+2*n]**2. / diag_of_cov_here) - 0.5 * log_det_cov_here * 2.0
        # the following makes sure that the number of active leaves stops at the maximum
        # else:
        #     break

    # print("max n", n)
    # breakpoint()
    # print(x_ML_1,model_1.fast_lnpost(jnp.array(x_ML_1)))
    # print(x_fixed,model_1.fast_lnpost(jnp.array(x_fixed)))
    inp = jnp.array(x_fixed)
    res = model_1.fast_lnlike(inp) + sum_lnprior_a
    if res == -jnp.inf:
        return -1e50
    return res

cov = {
    "powerlaw": np.diag(np.ones(ndims["powerlaw"])),
    "noise": np.diag(np.ones(ndims["noise"]))
}
for n in range(model_1_Nf):
    cov[f"coeff_{n}"] = np.diag(np.ones(ndims[f"coeff_{n}"]))

# import covariance
for br in branch_names:
    cov[br] = np.load(br+"_cov.npy") * 2.38**2 / ndims[br]

cov["noise"] = np.asarray([[cov["noise"]]])
# GammaInv = np.linalg.inv(Fisher_1.get_Fisher(x_ML_1)[2:-1,2:-1])
# cov["powerlaw"] = GammaInv[:2,:2]
# cov["powerlaw"] = GammaInv[2:,2:]
moves = [StretchMove()] # GaussianMove(cov), 

ensemble = EnsembleSampler(
    nwalkers,
    ndims,
    loglike,
    priors,
    tempering_kwargs=dict(ntemps=ntemps),#, Tmax=np.inf),
    nbranches=len(branch_names),
    branch_names=branch_names,
    nleaves_max=nleaves_max,
    nleaves_min=nleaves_min,
    backend="rj_results.h5",
    moves=moves,
    rj_moves=True,  # basic generation of new leaves from the prior
)

coords = {
    "powerlaw": np.zeros((ntemps, nwalkers, nleaves_max["powerlaw"], ndims["powerlaw"])),
    "noise": np.zeros((ntemps, nwalkers, nleaves_max["noise"], ndims["noise"])
    )
}
for n in range(model_1_Nf):
    coords[f"coeff_{n}"] = np.zeros((ntemps, nwalkers, nleaves_max[f"coeff_{n}"], ndims[f"coeff_{n}"]))

inds = {
    "noise": np.ones((ntemps, nwalkers, nleaves_max["noise"]), dtype=bool),
    "powerlaw": np.ones((ntemps, nwalkers, nleaves_max["powerlaw"]), dtype=bool)
}
for n in range(model_1_Nf):
    inds[f"coeff_{n}"] = np.zeros((ntemps, nwalkers, nleaves_max[f"coeff_{n}"]), dtype=bool)

# for n in range(model_1_Nf):
#     if n >= 2:
#         inds[f"coeff_{n}"] = np.random.choice([True, False], size=(ntemps, nwalkers, nleaves_max[f"coeff_{n}"]))
for nt in range(ntemps):
    for nw in range(nwalkers):
        n_max = np.random.randint(2, model_1_Nf+1)
        for n in range(0, n_max):
            inds[f"coeff_{n}"][nt, nw, 0] = True
        print([inds[f"coeff_{n}"][nt, nw, 0] for n in range(0, model_1_Nf)])
        # breakpoint()

# for each walker, only allow the first n coefficients to be active
# for nw in range(model_1_Nf):
#     for n in range(model_1_Nf):
#         if n >= nw+2:
#             inds[f"coeff_{n}"][:,nw] = False

# make sure to start near the proper setup
truths = {"powerlaw": hypers_inj, "noise": [sigma_inj]}
for n in range(model_1_Nf):
    # truths[f"coeff_{n}"] = a_ML_1[n*2:n*2+2]
    truths[f"coeff_{n}"] = a_inj[n*2:n*2+2]

for br in branch_names:
    print("starting close to", truths[br], "for branch", br)
    coords[br] = priors[br].rvs((ntemps, nwalkers, nleaves_max[br]))
    # walker 0 will start at close to the truth
    # coords[br][:ntemps//2, :nwalkers//2, 0] = np.random.multivariate_normal(truths[br], np.diag(np.ones(ndims[br]) * 0.01), size=(ntemps, nwalkers))[:ntemps//2, :nwalkers//2]
    coords[br][:, :, 0] = np.random.multivariate_normal(truths[br], np.diag(np.ones(ndims[br]) * 0.01), size=(ntemps, nwalkers))[:, :]

# check loglike around true
x_test = [np.asarray(truths[br]) for br in branch_names]
print("test:", loglike(x_test))

etas = ensemble.temperature_control.betas
log_prior = ensemble.compute_log_prior(coords, inds=inds)
log_like = ensemble.compute_log_like(coords, inds=inds, logp=log_prior)[0]

# make sure it is reasonably close to the maximum which this is
# will not be zero due to noise
print(log_like, log_prior)

# setup starting state
state = State(coords, log_like=log_like, log_prior=log_prior, inds=inds)
nsteps = 10_000
# run the MCMC
last_sample = ensemble.run_mcmc(state, nsteps, burn=1000, progress=True, thin_by=1)
print("acceptance rate: ", np.mean(ensemble.acceptance_fraction,axis=1), np.mean(ensemble.rj_acceptance_fraction,axis=1))
# plot loglike
loglike = ensemble.get_log_like()[:,0]
plt.figure()
plt.plot(loglike)
plt.xlabel('step')
plt.ylabel('loglike')
plt.savefig('rj_loglike.pdf')
labels = {"powerlaw": ["log_amp", "gamma"], "noise": ["log_sigma"]}
for n in range(model_1_Nf):
    labels[f"coeff_{n}"] = [f"a_{n}", f"b_{n}"]


stored_samples = {}

for br in branch_names:
    print("processing ", br)
    input_to_plot = []
    for nleaf in range(nleaves_max[br]):
        ind_tmp = ensemble.get_inds()[br][:,0,:,nleaf]
        # estimate how many ind_tmp are True and how many are False
        n_true = ind_tmp.sum()
        n_false = (~ind_tmp).sum()
        # print(f"n_true: {n_true}, n_false: {n_false}, ratio: {n_true/n_false}")
        if n_true == 0:
            continue
        samp_tmp = ensemble.get_chain()[br][:,0,:,nleaf]
        input_to_plot = samp_tmp[ind_tmp]
        # print(input_to_plot.shape, truths[br].shape)
        if br == "noise":
            tr = None
        else:
            tr = truths[br]

        stored_samples[br] = input_to_plot
        # add test plt.title(f"branch {br} BF: {n_true/n_false}");
        fig = corner.corner(input_to_plot, 
                    labels=labels[br], 
                    truths=tr,
                    bins=40,
                    levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
                    plot_density=False,
                    plot_datapoints=False,
                    fill_contours=False,
                    show_titles=False,
                    # max_n_ticks=3,
                    # truth_color='k',
                    # labelpad=0.3,
                    )
        plt.title(f"BF: {n_true/n_false:.2e}");
        plt.tight_layout()
        plt.savefig("rj_" +br + f'_nleaf{nleaf}_corner.pdf'); plt.close('all')

# bns = (
#     np.arange(0, nleaves_max["coeff"] + 2) - 0.5
# )  # Just to make it pretty and center the bins

# fig, ax = plt.subplots(1, 1, sharex=True)#, figsize=(10, 10))
# for temp, ax_t in enumerate([ax]):
#     inds_here = ensemble.get_inds()["coeff"][:, temp] 
#     # bayes factor for the presence of a signal
#     BF = np.sum(inds_here)/np.sum(~inds_here)
#     # set title for each subplot with the temperature and the bayes factor
#     ax_t.hist(np.sum(inds_here,axis=-1).flatten(), bins=bns, density=True, rwidth=0.8)
#     # BF 1 vs 0
#     den = (np.sum(inds_here,axis=-1).flatten() == 0).sum()
#     if den == 0.:
#         den = 1.0
#     BF = [(np.sum(inds_here,axis=-1).flatten() == ii).sum() / den for ii in range(1,nleaves_max["coeff"]+1)]
#     BF_str = ", ".join([f"BF {ii}vs0: {BF[ii-1]:.1e}" for ii in range(1, nleaves_max["coeff"] + 1)])
#     print(BF_str)
#     ax_t.set_title(f"Temperature {1/betas[temp]:.1e} - {BF_str}")
#     print(f"Temperature {1/betas[temp]:.3}")
# ax.set_xlabel("Number of Signals")
# plt.tight_layout()
# plt.savefig('nleaves_hist.pdf')
