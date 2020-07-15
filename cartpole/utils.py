import numpy as np
import gym
import tensorflow as tf
from random import shuffle
import math
from pdb import set_trace

def generate_batch(env, env_name, pi, num_trajs):
    print ('generating batch for env {}'.format(env_name))
    trajs = []
    while len(trajs) < num_trajs:
        state, acc_r, done = env.reset(), 0., False
        state_f = env.state
        traj = []
        while True:
            a = pi.action(state)
            n_state, r, done, info = env.step(a)
            traj.append((state, a, r, n_state, state_f))
            state = n_state
            state_f = env.state
            acc_r += r
            if env_name == 'MountainCar-v0':
                if state[0] >= 0.5:
                    break
            elif env_name == 'CartPole-v0':
                # termination condition for CartPole-v0
                if abs(state[2]) > 12 or abs(state[0]) > 2.4:
                    break
            elif env_name == 'Acrobot-v1':
                if done:
                    break
        trajs.append(traj)
    return trajs

def sample_states(trajs, num_states):
    states = np.array([(s[0], s[4])for traj in trajs for s in traj])
    idx = np.random.choice(states.shape[0], num_states, replace = False)
    return states[idx, :]

def filter_bad_states(env, env_name, pi, states):

    if env_name != 'Acrobot-v1':
        return states

    filtered_states = []
    uw_env = env.unwrapped
    for s in states:
        state, acc_r, done = uw_env.reset(), 0., False
        # setting initial state to be state to be evaluated
        if env_name == 'Acrobot-v1':
            uw_env.state = np.array(s[1])
            state = np.array(s[0])
        t = 0
        bad_state = False
        while True:
            a = pi.action(state)
            n_state, r, done, info = uw_env.step(a)
            state = n_state
            acc_r += r
            t += 1
            if env_name == 'Acrobot-v1':
                if t >= 300:
                    bad_state = True
                    break
                if done:
                    break
        if not bad_state:
            filtered_states.append((s[0], s[1]))

    print ('number of states removed {}'.format(len(states) - len(filtered_states)))
    return filtered_states

def mc_rollouts(env, env_name, pi, num_rolls, states):

    print ('generating MC rollouts for env {}'.format(env_name))
    uw_env = env.unwrapped
    vf = {}
    st_count = 0
    for s in states:
        vals = []
        rolls = 0 
        while rolls < num_rolls:
            state, acc_r, done = uw_env.reset(), 0., False
            # setting initial state to be state to be evaluated

            if env_name == 'Acrobot-v1':
                uw_env.state = np.array(s[1])
                #st = uw_env.state
                #state = np.array([cos(st[0]), sin(st[0]), cos(st[1]), sin(st[1]), st[2], st[3]])
                state = np.array(s[0])
            elif env_name == 'CartPole-v0' or env_name == 'MountainCar-v0':    
                uw_env.state = np.array(s[0])
                state = uw_env.state
            t = 0
            while True:
                a = pi.action(state)
                n_state, r, done, info = uw_env.step(a)
                state = n_state
                acc_r += r
                t += 1
                if env_name == 'MountainCar-v0':
                    if state[0] >= 0.5:
                        break
                elif env_name == 'CartPole-v0':
                    # termination condition for CartPole-v0
                    if abs(state[2]) > 12 or abs(state[0]) > 2.4:
                        break
                elif env_name == 'Acrobot-v1':
                    if done or t >= 150:
                        break
            #print ('roll traj len {}'.format(t))
            vals.append(acc_r)
            rolls += 1
            #print ('finished roll num {}'.format(rolls))
        #print (vals)
        st_count += 1
        print ('{} finished rolls for state'.format(st_count))
        vf[tuple(s[0])] = np.mean(vals)
    return vf

def mse(vf_1, vf_2):
    assert len(vf_1) == len(vf_2)
    err = 0.
    for i in range(len(vf_1)):
        err += (vf_1[i] - vf_2[i]) ** 2.
    err /= len(vf_1)
    return err

def load_IWs(trajs, pi, pi_b):
    data = []
    rhos = []
    for traj in trajs:
        temp = []
        for idx, step in enumerate(traj):
            # s must be raw state
            s = step[0]
            a = step[1]
            r = step[2]
            n_s = step[3]
            rho = pi.action_prob(s, a) / pi_b.action_prob(s, a)
            temp.append((s, a, r, n_s, rho))
            rhos.append(rho)
        data.append(temp)
    print ('mean rhos {}'.format(np.mean(rhos)))
    print ('std rhos {}'.format(np.std(rhos)))
    print ('max rhos {}'.format(np.max(rhos)))
    print ('min rhos {}'.format(np.min(rhos)))
    return data

def pre_process_batch(env, pi, trajs, V):

    data_x = []
    data_y = []

    for traj in trajs:
        temp = []
        for idx, step in enumerate(traj):
            s = step[0]
            a = step[1]
            temp_a = np.zeros((env.action_space.n))
            # probability dist over actions for a given state, instead of the one-hot
            # encoding of the action taken
            if pi is not None:
                for i in range(env.action_space.n):
                    temp_a[i] = pi.action_prob(s, i)
            else:
                temp_a[a] = 1.

            assert abs(sum(temp_a) - 1.) <= 1e-2
            if V is not None:
                s = V.get_feature_vec(s)#[0]
            data_x.append(s)
            data_y.append(temp_a)
    return np.array(data_x), np.array(data_y)

def split_data(data_x, data_y, ratio = 0.1):

    if ratio == 0.:
        return data_x, data_y, [], []

    num_samples = len(data_x) - 1
    val_inds = sorted(np.random.choice(num_samples,
        int(math.floor(ratio * num_samples)), replace = False))
    tr_inds = [x for x in range(num_samples) if x not in val_inds]

    tr_inds_set = set(tr_inds)
    val_inds_set = set(val_inds)

    assert len(tr_inds_set) == len(tr_inds)
    assert len(val_inds_set) == len(val_inds)
    assert len(tr_inds_set.intersection(val_inds)) == 0

    shuffle(tr_inds)
    shuffle(val_inds)

    tr_inds = np.array(tr_inds)
    val_inds = np.array(val_inds)

    train_x = data_x[tr_inds]
    train_y = data_y[tr_inds]
    test_x = data_x[val_inds]
    test_y = data_y[val_inds]

    assert len(train_x) == len(train_y)
    assert len(train_x) == len(tr_inds)
    assert len(test_x) == len(test_y)
    assert len(test_y) == len(val_inds)

    return train_x, train_y, test_x, test_y

