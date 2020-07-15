import numpy as np
from policy import Policy
import utils
import time
from pdb import set_trace

class ValueFunctionWithApproximation(object):
    def __call__(self,s) -> float:
        """
        return the value of given state; \hat{v}(s)

        input:
            state
        output:
            value of the given state
        """
        raise NotImplementedError()

    def update(self,alpha,G,s_tau):
        """
        Implement the update rule;
        w <- w + \alpha[G- \hat{v}(s_tau;w)] \nabla\hat{v}(s_tau;w)

        input:
            alpha: learning rate
            G: TD-target
            s_tau: target state for updating (yet, update will affect the other states)
        ouptut:
            None
        """
        raise NotImplementedError()

def semi_gradient_n_step_td_batch(
    gamma:float,
    n:int,
    alpha:float,
    V:ValueFunctionWithApproximation,
    batch:np.array,
    sampled_states:np.array,
    true_vf = None
):
    mod_batch = []
    sps = []
    flag = []
    for traj in batch:
        for idx, step in enumerate(traj):
            s, a, r, s_prime, rho = step
            # if we should bootstrap on next state
            next_bootstrap = idx < len(traj) - 1   
            mod_batch.append([s, a, r, rho])
            sps.append(s_prime)
            flag.append(next_bootstrap)

    errors = []
    old_vf = [V(s)[1] for s in sampled_states]
    p = 0
    print ('new algo method')
    while True:
        next_s_vals = V.get_curr_val(sps)
        next_s_vals = np.array([val if flag[idx] else 0 for idx, val in enumerate(next_s_vals)])
        temp_mod_batch = np.array([[step[0], step[1], step[2] + next_s_vals[idx], step[3]] for idx, step in enumerate(mod_batch)])

        V.accum_batch(alpha, temp_mod_batch[:, 2], temp_mod_batch[:, 0], temp_mod_batch[:, 1], temp_mod_batch[:, 3])
        V.update()
        V.flush_accum()
        
        p += 1
        new_vf = [V(s)[1] for s in sampled_states]

        temp_vf = np.array([abs(v) for v in new_vf])
        diverged = np.any(temp_vf >= 1e4) or np.any(np.isnan(temp_vf))

        '''
        if p % 50  == 0:
            temp_new_vf = np.round(new_vf, 2)
            errors.append(utils.mse(true_vf, new_vf)) 
            #print ('batch process step {},  vf {}'.format(p, temp_new_vf))
            if true_vf is not None:
                print ('mse {} tr cost (before update, check error below) {}'.format(utils.mse(true_vf, new_vf), V.temp_cost))
        '''

        if diverged:
            break

        c = 0
        for i in range(len(sampled_states)):
            if abs(old_vf[i] - new_vf[i]) <= 1e-1:
                c += 1
        if c == len(sampled_states):
            break
        old_vf = new_vf

    return errors

