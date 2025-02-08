'''Script does Reversible-jump MCMC.'''



from jax import jit, vmap
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from BayesFT.chains import Chain




# Parallel tempering swap
def PT_swap(num_chains,
            temp_ladder,
            lnpost_func,
            accept_counts,
            reject_counts,
            chains,
            keys):

    # track swaps
    states = [jnp.copy(chain.state) for chain in chains]
    swap_map = list(range(num_chains))

    # loop through and propose a swap at each chain (starting from hottest chain and going down in T)
    # and keep track of results in swap_map
    for j, swap_chain in enumerate(reversed(range(num_chains - 1))):
        log_acc_ratio = -lnpost_func(states[swap_map[swap_chain]], temperature=temp_ladder[swap_chain])
        log_acc_ratio += -lnpost_func(states[swap_map[swap_chain + 1]], temperature=temp_ladder[swap_chain + 1])
        log_acc_ratio += lnpost_func(states[swap_map[swap_chain + 1]], temperature=temp_ladder[swap_chain])
        log_acc_ratio += lnpost_func(states[swap_map[swap_chain]], temperature=temp_ladder[swap_chain + 1])
        acc_ratio = np.exp(log_acc_ratio)
        
        # accept or reject swap
        if jr.uniform(keys[j]) <= acc_ratio:
            swap_map[swap_chain], swap_map[swap_chain + 1] = swap_map[swap_chain + 1], swap_map[swap_chain]
            accept_counts[-1, swap_chain] += 1
        else:
            reject_counts[-1, swap_chain] += 1

    # loop through the chains and record the new samples and likelihood values
    for j in range(num_chains):
        chains[j].add_sample(states[swap_map[j]], lnpost_func(states[swap_map[j]], temperature=temp_ladder[j]))
    
    return


def RJMCMC(num_samples,
           num_chains,
           x0,
           lnpost_func,
           jump_proposals,
           temp_ladder=None,
           PT_weight=20):

    # if no temperature ladder provided, use geometric spacing
    chain_ndx = np.arange(num_chains)
    if temp_ladder is None:
        c = 1.1
        temp_ladder = c**chain_ndx

    # vectorize posterior function
    vectorized_lnpost = jit(vmap(lnpost_func, in_axes=(0, 0)))

    # initialize chains
    lnposts0 = vectorized_lnpost(np.tile(x0, (num_chains, 1)), temp_ladder)
    chains = [Chain(x0, lnpost0, temp)
              for lnpost0, temp in zip(lnposts0, temp_ladder)]

    # organize jump proposals
    num_jump_types = len(jump_proposals)
    jump_functions = []
    jump_names = []
    jump_weights = []
    for proposal in jump_proposals:
        jump_function, weight = proposal
        jump_functions.append(jump_function)
        jump_names.append(jump_function.__name__)
        jump_weights.append(weight)
    # add PT jump proposal
    num_jump_types += 1
    jump_names.append('PT swap')
    jump_weights.append(PT_weight)

    # make jump choices
    jump_selections = np.random.choice(num_jump_types, num_samples, p=jump_weights/np.sum(jump_weights))

    # track jump acceptance
    accept_counts = np.zeros((num_jump_types, num_chains))
    reject_counts = np.zeros((num_jump_types, num_chains))

    # main MCMC loop
    for i in range(num_samples - 1):

        # update progress
        if i % (num_samples // 1000) == 0:
            print(f'{round(i / num_samples * 100, 3)}%', end='\r')

        # index for jump choice
        jump_ndx = jump_selections[i]

        # independent random keys for chains
        keys = jr.split(jr.PRNGKey(i), num_chains)

        # jump proposals
        if jump_ndx == num_jump_types - 1 and PT_weight != 0:  # parallel tempering swap
            
            PT_swap(num_chains,
                    temp_ladder,
                    lnpost_func,
                    accept_counts,
                    reject_counts,
                    chains,
                    keys)
        else:

            # proposal method
            jump_function = jump_functions[jump_ndx]

            for j, (chain, key) in enumerate(zip(chains, keys)):
                # propose jump for chain
                new_state, new_lnpost, accept_prob = jump_function(chain, key)
                
                # decide whether or not to accept
                if jr.uniform(key) < accept_prob:  # accept
                    accept_counts[jump_ndx, j] += 1
                    chain.add_sample(new_state, new_lnpost)
                else:
                    reject_counts[jump_ndx, j] += 1
                    chain.add_sample(chain.state, chain.lnpost)
    

    # compute jump proposal acceptance rates
    acceptance_rates = accept_counts / (accept_counts + reject_counts)
    print('Jump acceptance rates')
    for name, rate in zip(jump_names, acceptance_rates):
        print(f'{name}: {rate}')

    return chains