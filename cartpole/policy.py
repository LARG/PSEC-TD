import numpy as np
import gym
from policy_struct import NN
from pdb import set_trace

class Policy(object):
    def __init__(self, env):
        pass

    def action(self, state):
        pass

class BalancePolicy(Policy):
    def __init__(self, env, stoch = 0.5):
        self.env = env
        self.stoch = stoch

        self.obs_dim = env.observation_space.shape[0] 
        self.num_actions = self.env.action_space.n
        prob = self.stoch * (1. / self.num_actions)
        probs_vel_neg = [prob for _ in range(self.num_actions)]
        probs_vel_pos = [prob for _ in range(self.num_actions)]

        probs_vel_neg[0] += (1. - self.stoch)
        probs_vel_pos[1] += (1. - self.stoch)

        assert abs(sum(probs_vel_neg) - 1.) <= 1e-2
        assert abs(sum(probs_vel_pos) - 1.) <= 1e-2
        self.probs_vel_neg = probs_vel_neg
        self.probs_vel_pos = probs_vel_pos

    def action(self, state):

        assert len(state) == self.obs_dim
        prob = []
        if state[2] < 0:
            prob = self.probs_vel_neg
        else:
            prob = self.probs_vel_pos
        return np.random.choice(self.num_actions, size = 1, p = prob, replace = False)[0]
        
    def action_prob(self, state, act):
        assert len(state) == self.obs_dim
        if state[2] < 0:
            return self.probs_vel_neg[act]
        else:
            return self.probs_vel_pos[act]
    
    def action_probs(self, state):
        assert len(state) == self.obs_dim
        if state[2] < 0:
            return self.probs_vel_neg
        else:
            return self.probs_vel_pos

    def action_probs_batch(self, s):
        assert len(s[0]) == self.obs_dim
        probs = []
        for i in s:
            if i[2] < 0:
                probs.append(self.probs_vel_neg)
            else:
                probs.append(self.probs_vel_pos)
        return np.array(probs)

class NNPolicy(Policy):
    def __init__(self, env, num_hidden_layers, num_neurons, act_fn, ckpt = False, opt = 'gd', scope_name = 'pol'):
        self.env = env
        ckpt_split = ckpt.split('/')
        ckpt = ('/'.join(ckpt_split + [ckpt_split[1]])) + '-0'
        lr = 0
        print (ckpt)
        self.pi = NN(env.observation_space.shape[0], env.action_space.n, num_hidden_layers, num_neurons, act_fn, lr, ckpt = ckpt, opt = opt, scope_name = scope_name)

    def action(self,s):
        s = np.reshape(s, (1, len(s)))
        return self.pi.action(s)

    def action_prob(self, s, a):
        s = np.reshape(s, (1, len(s)))
        return self.pi.action_prob(s, a)

    def action_probs(self, s):
        s = np.reshape(s, (1, len(s)))
        return self.pi.action_probs(s)

    def action_probs_batch(self, s):
        return self.pi.action_probs_batch(s)

class LinearPolicy(Policy):
    def __init__(self, env, weightfile = None, V = None):
        self.env = env
        self.weightfile = weightfile

        # used for tilecoding
        self.V = V

        # TODO maybe use protobuf, and include tilecoding information in the original
        # training pb instead of redefining again externally and passing it in
        if self.weightfile is not None:
            self._load_weights()

    def _load_weights(self):
        self.weights = []
        f = open(self.weightfile, 'r')
        for line in f:
            line_cmps = line.split(',')
            if len(line_cmps) == self.env.action_space.n:
                temp = [float(i) for i in line_cmps]
                self.weights.append(temp)

        # last line is bias, make it bias, and remove from weight array
        self.bias = np.array(self.weights[-1])
        self.weights = np.array(self.weights[:-1])
        print ('loaded weights with dimensions')
        # should be feature_vec_len * num_actions
        print ('weights shape {}'.format(self.weights.shape))
        print ('bias shape {}'.format(self.bias.shape))
    
    def action(self, state):
        # get feature vector
        s = state
        if self.V is not None:
            s = self.V(state)[0]
        out = np.matmul(s, self.weights) + self.bias
        out = np.exp(out) / np.sum(np.exp(out), keepdims = True)
        assert len(out) == self.env.action_space.n
        assert abs(sum(out) - 1.) <= 1e-2
        return np.random.choice(out.shape[0], size = 1, p = out, replace = False)[0]

    def action_prob(self, state, act):
        s = state
        if self.V is not None:
            s = self.V(state)[0]
        out = np.matmul(s, self.weights) + self.bias
        out = np.exp(out) / np.sum(np.exp(out), keepdims = True)
        assert len(out) == self.env.action_space.n
        assert abs(sum(out) - 1.) <= 1e-2
        return out[act]

