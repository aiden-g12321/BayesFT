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
data_analyzed = signal_inj + noise

from eryn.backends import HDFBackend
import corner
import matplotlib.pyplot as plt 
import numpy as np
data = HDFBackend("rj_results.h5")
log_amp_inj = 1.4
gamma_inj = 3.3
hypers_inj = np.array([log_amp_inj, gamma_inj])
sigma_inj = 1.0

truths = {"powerlaw": hypers_inj, "noise": [sigma_inj]}
for n in range(10):
    # truths[f"coeff_{n}"] = a_ML_1[n*2:n*2+2]
    truths[f"coeff_{n}"] = a_inj[n*2:n*2+2]


branch_names = data.branch_names
nleaves_max = data.nleaves_max

# plot loglike
loglike = data.get_log_like()[:,0]
plt.figure()
plt.plot(loglike)
plt.xlabel('step')
plt.ylabel('loglike')
plt.savefig('loglike.pdf')

labels = {"powerlaw": ["log_amp", "gamma"], "noise": ["log_sigma"]}
for n in range(10):
    labels[f"coeff_{n}"] = [f"a_{n}", f"b_{n}"]

discard = int(data.iteration*0.25)

cov = {}
for br in branch_names:
    print("processing ", br)
    input_to_plot = []
    nleaf = 0
    ind_tmp = data.get_inds(discard=discard)[br][:,0,:,nleaf]
    # estimate how many ind_tmp are True and how many are False
    n_true = ind_tmp.sum()
    n_false = (~ind_tmp).sum()
    print(f"n_true: {n_true}, n_false: {n_false}, ratio: {n_true/n_false}")
    if n_true == 0:
        continue
    samp_tmp = data.get_chain(discard=discard)[br][:,0,:,nleaf]
    
    input_to_plot = samp_tmp[ind_tmp]
    np.save(br+"_cov.npy", np.cov(input_to_plot, rowvar=False))
    
    # print(input_to_plot.shape, truths[br].shape)
    if br == "noise":
        tr = None
    else:
        tr = truths[br]

    fig = corner.corner(input_to_plot, 
                labels=labels[br], 
                truths=tr,
                bins=40,
                levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
                plot_density=False,
                plot_datapoints=False,
                fill_contours=False,
                show_titles=False,
                )
    plt.title(f"BF: {n_true/n_false:.2e}");
    plt.tight_layout()
    plt.savefig("rj_" +br + f'_nleaf{nleaf}_corner.pdf'); plt.close('all')


a_vec = np.zeros((2*model_1_Nf, samp_tmp.shape[0], data.ntemps, data.nwalkers))
f_vec = np.zeros((model_1_Nf, samp_tmp.shape[0], data.ntemps, data.nwalkers))
coeff_br = [br for br in branch_names if "coeff" in br]
ind_true = np.array([data.get_inds(discard=discard)[br][...,0] for br in coeff_br])
ind_true_a = np.repeat(np.array([data.get_inds(discard=discard)[br][...,0] for br in coeff_br]),2,axis=0)

for ii,br in enumerate(coeff_br):
    ind_tmp = data.get_inds(discard=discard)[br][:,:,:,0]
    samp_tmp = data.get_chain(discard=discard)[br][:,:,:,0]
    a_vec[2*ii:2*(ii+1)][0] = samp_tmp[...,0]
    a_vec[2*ii:2*(ii+1)][1] = samp_tmp[...,1]
    f_vec[ii] = f_inj[ii]

it, nt, nw = 0, 0, 0
f_array = np.array(f_inj)
result_per_freq = {f_array [ii]: [] for ii in range(model_1_Nf)}
plt.figure(figsize=(12, 5))
plt.scatter(t, data_analyzed, color='C0', s=10, label='data')
for it in range(0, samp_tmp.shape[0], 500):
    for nw in range(data.nwalkers):
        a_temp = a_vec[:,it,nt,nw][ind_true_a[:,it,nt,nw]]
        f_temp = f_vec[:,it,nt,nw][ind_true[:,it,nt,nw]]
        logPSD_samples = np.log(a_temp[::2]**2. + a_temp[1::2]**2.)
        F = f.fast_Fourier_design(t, f_temp)
        signal = F @ a_temp
        for k,ff in enumerate(f_temp):
            result_per_freq[ff].append(logPSD_samples[k])
        plt.plot(t, signal,  color='grey', alpha=0.1)
plt.xlabel('time [s]')
plt.ylabel('signal [units]')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('rj_signal.pdf')

import matplotlib.patches as mpatches

labels = []
def add_label(violin, label):
    color = violin["bodies"][0].get_facecolor().flatten()
    labels.append((mpatches.Patch(color=color), label))

# create violin plot for each frequency
plt.figure()
add_label(plt.violinplot(result_per_freq.values(), positions=result_per_freq.keys(), showmeans=True, showextrema=False), 'RJ-MCMC samples')
add_label(plt.violinplot(np.log(many_a[:,::2]**2 + many_a[:,1::2]**2)[:,:model_1_Nf], positions=f_array[:model_1_Nf], showmeans=True, showextrema=False), 'Realizations from power law')

plt.semilogx(f_array[:model_1_Nf], np.log(a_inj[::2]**2 + a_inj[1::2]**2)[:model_1_Nf], '-o',color='red', label='Injected')
labels.append((mpatches.Patch(color='red'), 'Injected'))

plt.legend(*zip(*labels), loc='lower left')
plt.xlabel('frequency [Hz]')
plt.ylabel('logPSD')
plt.tight_layout()
plt.savefig('rj_violin.pdf')
plt.show()
