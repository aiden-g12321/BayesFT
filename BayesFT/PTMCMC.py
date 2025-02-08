'''Script does parallel tempering MCMC.'''


from jax import jit, vmap
import jax.numpy as jnp
import jax.random as jr
import numpy as np




# Parallel tempering swap
def PT_swap(num_chains,
            temp_ladder,
            lnpost_func,
            iteration,
            PT_accept_counts,
            PT_reject_counts,
            samples,
            lnposts,
            keys):

    # track swaps
    swap_map = list(range(num_chains))
    states = samples[:, iteration]

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
            PT_accept_counts[-1, swap_chain] += 1
        else:
            PT_reject_counts[-1, swap_chain] += 1

    # loop through the chains and record the new samples and likelihood values
    for j in range(num_chains):
        samples[j, iteration + 1] = states[swap_map[j]]
        # lnposts[j, iteration + 1] = lnposts[swap_map[j], iteration] * temp_ladder[swap_map[j]] / temp_ladder[j]
        lnposts[j, iteration + 1] = lnpost_func(samples[j, iteration + 1], temperature=temp_ladder[j])

    return



def PTMCMC(num_samples,
           num_chains,
           x0,
           lnpost_func,
           jump_proposals,
           temp_ladder=None,
           PT_weight=20):

    # if no temperature ladder provided, use geometric spacing
    chain_ndx = np.arange(num_chains)
    if temp_ladder is None:
        c = 1.4
        temp_ladder = c**chain_ndx

    # vectorize posterior function
    vectorized_lnpost = jit(vmap(lnpost_func, in_axes=(0, 0)))

    # initialize samples
    ndim = x0.shape[0]
    samples = np.zeros((num_chains, num_samples, ndim))
    lnposts = np.zeros((num_chains, num_samples))
    samples[:, 0] = np.tile(x0, (num_chains, 1))
    lnposts[:, 0] = np.array([lnpost_func(samples[j, 0], temperature=temp)
                              for j, temp in zip(chain_ndx, temp_ladder)])

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
    
    # function to decide acceptance / rejection
    def accept_reject(new_state, new_lnpost, accept_prob, prev_state, prev_lnpost, key):
        accept = jr.uniform(key) < accept_prob
        final_state = jnp.where(accept, jnp.copy(new_state), jnp.copy(prev_state))
        final_lnpost = jnp.where(accept, new_lnpost, prev_lnpost)
        return final_state, final_lnpost, accept
    
    # vectorized acceptance / rejection
    vec_accept_reject = jit(vmap(accept_reject, in_axes=(0, 0, 0, 0, 0, 0)))

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
        if jump_ndx == num_jump_types - 1 and PT_weight != 0:  # If Parallel Tempering Swap is selected
            PT_swap(num_chains,
                    temp_ladder,
                    lnpost_func,
                    i,
                    accept_counts,
                    reject_counts,
                    samples,
                    lnposts,
                    keys)
        else:
            # Select jump function before applying JAX transformations
            vectorized_jump_function = jump_functions[jump_ndx]
            # Propose jumps
            new_states = vectorized_jump_function(samples[chain_ndx, i],
                                                               i,
                                                               temp_ladder,
                                                               keys)
            # print(new_states[0,0] - samples[0,i, 0])
            
            # evaluate posterior at new points
            new_lnposts = vectorized_lnpost(new_states, temp_ladder)

            # acceptance probabilities
            accept_probs = jnp.exp(new_lnposts - lnposts[chain_ndx, i])

            # accept or reject proposal
            final_states, final_lnposts, accepted = vec_accept_reject(new_states,
                                                                    new_lnposts,
                                                                    accept_probs,
                                                                    samples[chain_ndx, i],
                                                                    lnposts[chain_ndx, i],
                                                                    keys)

            # convert back to NumPy and update
            samples[chain_ndx, i + 1] = jnp.asarray(final_states)
            lnposts[chain_ndx, i + 1] = jnp.asarray(final_lnposts)

            # Update acceptance/rejection counts
            accept_counts[jump_ndx, chain_ndx] += jnp.asarray(accepted, dtype=int)
            reject_counts[jump_ndx, chain_ndx] += jnp.asarray(1 - accepted, dtype=int)

    # compute jump proposal acceptance rates
    acceptance_rates = accept_counts / (accept_counts + reject_counts)
    print('Jump acceptance rates')
    for name, rate in zip(jump_names, acceptance_rates):
        print(f'{name}: {rate}')

    return samples, lnposts, temp_ladder