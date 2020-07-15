'''
Main script that does the following:
- computes true VF values by runnning Monte Carlo runs
- computes PSEC MLE policy (PSEC and RIS are used interchangeably)
- computes VF with TD or PSEC-TD
- stores error in a protobuf file

This file is typically executed by a runner file as described in the README
'''

import numpy as np
import gym
from utils import generate_batch, sample_states, mc_rollouts, mse, load_IWs, filter_bad_states
from vf_nn import ValueFunctionWithNN
from algo import semi_gradient_n_step_td_batch
from ris_policy import RISNNPolicy
from policy import BalancePolicy, NNPolicy
import argparse
from protos import results_pb2
import tensorflow as tf

parser = argparse.ArgumentParser()

parser.add_argument('--env_name', type = str, help='environment name', default = None)
parser.add_argument('--outfile', type = str, help='saving result file', default = None)
parser.add_argument('--seed', type = int, help='seed for randomness')

parser.add_argument('--algo', type = str, default = 'td', help='td')
parser.add_argument('--ris', default = False, action='store_true', help='if applying PSEC')
parser.add_argument('--num_trajs', default = 50, type=int, help='num trajs for batch')

# evaluation/behavior policy to execute
parser.add_argument('--pi', default = 0, type=int, help='2/3 for neural net')
parser.add_argument('--pi_b', default = 0, type=int, help=' 2/3 for neural net')
parser.add_argument('--pi_ris', default = 0, type=int, help='3 for neural net')
parser.add_argument('--pi_weightfile', default = str, help='weight file for evaluation policy')
parser.add_argument('--pi_b_weightfile', default = str, help='weight file for behavior policy')
parser.add_argument('--pi_b_psec_weightfile', type=str, default = None, help='same value as pi_b_weightfile but distinct since PSEC version may be tuned')

# NN PSEC specs
parser.add_argument('--psec_alpha', default = 0, type=float, help='learning rate for PSEC MLE policy')
parser.add_argument('--psec_num_neurons', default = 16, type=int, help='number of neurons per layer for PSEC MLE policy')
parser.add_argument('--psec_num_hidden_layers', default = 2, type=int, help='number of layers for PSEC MLE policy')
parser.add_argument('--psec_act_fn', default = 0, type=int, help='activation functino used for PSEC MLE policy')
parser.add_argument('--psec_linear_finetune', default = False, action='store_true', help='if tuning only last linear layer of PSEC policy')
parser.add_argument('--psec_nn_finetune', default = False, action='store_true', help='if finetuning whole PSEC network')

# MC rollout related for true VF comparison
parser.add_argument('--num_mc_trajs', default = 300, type=int, help='The number of trajs to collect before sampling MC states.')
parser.add_argument('--num_mc_sampled_states', default = 200, type=int, help='The number of states to sampled for MC rollouts.')
parser.add_argument('--num_mc_rollouts', default = 300, type=int, help='The number of MC rollouts to determine "true" VF')

# MDP specifics for training
parser.add_argument('--alpha', default = 0, type=float, help='VF learning rate')
parser.add_argument('--gamma', default = 1, type=float, help='discount factor')
parser.add_argument('--n', default = 1, type=int, help='n-step (RIS) TD. In this paper, n = 1 always')

# vf function specifics
parser.add_argument('--vf_rep', type = str, default = 'mlp', help='function approximator for value function')
parser.add_argument('--vf_num_neurons', default = 32, type=int, help='number of neurons per layer in VF')
parser.add_argument('--vf_num_hidden_layers', default = 2, type=int, help='number of hidden layers in VF')
parser.add_argument('--vf_act_fn', default = 0, type=int, help='activation function for VF')

FLAGS = parser.parse_args()

def set_global_seed(seed):
    tf.set_random_seed(seed)
    np.random.seed(seed)

if  __name__ == "__main__":
    env_name = FLAGS.env_name
    env = gym.make(env_name)
    env.seed(FLAGS.seed)
    set_global_seed(FLAGS.seed)
    print (FLAGS)

    results = results_pb2.MethodResult()

    # setting evaluation policy
    if FLAGS.pi == 0:
        stoch = 0.6
        if env_name == 'CartPole-v0':
            pi = BalancePolicy(env, stoch = stoch)
            print ('evaluation policy {} (Balance), p {}'.format(FLAGS.pi, stoch))
    elif FLAGS.pi == 2:
        print ('------ trying to load eval 2 --------')
        pi = NNPolicy(env, 2, 64, 0, ckpt = FLAGS.pi_weightfile, scope_name = 'eval_pol')
        print ('evaluation policy {} (NNPolicy), ckpt {}'.format(FLAGS.pi, FLAGS.pi_weightfile))
    elif FLAGS.pi == 3:
        print ('------ trying to load eval 3 --------')
        #pi = NNPolicy(env, 2, 16, 0, opt = 'gd', ckpt = FLAGS.pi_weightfile, scope_name = 'pgpol')
        pi = NNPolicy(env, 2, 16, 0, opt = 'adam', ckpt = FLAGS.pi_weightfile, scope_name = 'pgpol')
        print ('evaluation policy {} (NNPolicy), ckpt {}'.format(FLAGS.pi, FLAGS.pi_weightfile))

    # setting behavior policy
    if FLAGS.pi_b == 0:
        stoch_b = 0.6
        if env_name == 'CartPole-v0':
            pi_b = BalancePolicy(env, stoch = stoch)       
            print ('behavior policy {} (Balance), p {}'.format(FLAGS.pi_b, stoch_b))
    elif FLAGS.pi_b == 2:
        print ('------ trying to load beh --------')
        pi_b = pi
        print ('behavior policy {} (NNPolicy), ckpt {}'.format(FLAGS.pi_b, FLAGS.pi_b_weightfile))
    elif FLAGS.pi_b == 3:
        print ('------ trying to load beh --------')
        pi_b = pi
        print ('behavior policy {} (NNPolicy), ckpt {}'.format(FLAGS.pi_b, FLAGS.pi_b_weightfile))

    num_mc_trajs = FLAGS.num_mc_trajs
    num_mc_sampled_states = FLAGS.num_mc_sampled_states
    num_mc_rollouts = FLAGS.num_mc_rollouts

    # evaluating "true" value of certain states based on evaluation policy
    # generate trajectories to sample states
    mc_trajs = generate_batch(env, env_name, pi, num_mc_trajs)
    # sample states to compute MC estimates for
    mc_sampled_states = sample_states(mc_trajs, num_mc_sampled_states)

    mc_sampled_states = filter_bad_states(env, env_name, pi, mc_sampled_states)

    assert len(mc_sampled_states[0][0]) == env.observation_space.shape[0]
    # get MC estimates for sampled states (as "true" value)
    vf = mc_rollouts(env, env_name, pi, num_mc_rollouts, mc_sampled_states)

    # removing the full state
    mc_sampled_states = [s[0] for s in mc_sampled_states]

    #print (vf)
    # generate batch of data for actual (RIS)-TD(0) with behavior policy
    num_batch_trajs = FLAGS.num_trajs
    batch_raw = generate_batch(env, env_name, pi_b, num_batch_trajs)
    
    # intialize linear approximator using White settings (tile coding etc)
    # learn weights of linear approximator
    gamma = FLAGS.gamma
    n = FLAGS.n
    alpha = FLAGS.alpha

    ris = FLAGS.ris 

    if ris:
        val_batch_raw = generate_batch(env, env_name, pi_b, max(int(0.1 * num_batch_trajs), 1))
        psec_save_dir = None
        if FLAGS.outfile is not None:
            psec_save_dir = FLAGS.outfile.split('/')[2]
        # used ONLY for tc not for any approximation
        if FLAGS.pi_ris == 3:
            print ('RIS NN policy')
            preload_dir = None
            if (FLAGS.psec_linear_finetune or FLAGS.psec_nn_finetune) and (FLAGS.pi_b_psec_weightfile is not None):
                preload_dir = FLAGS.pi_b_psec_weightfile
            print ('preload RIS dir {}'.format(preload_dir))
            pi_ris = RISNNPolicy(env, batch_raw, val_batch_raw,\
                                    FLAGS.psec_num_hidden_layers,\
                                    FLAGS.psec_num_neurons,\
                                    FLAGS.psec_act_fn,\
                                    FLAGS.psec_alpha,\
                                    save_dir = psec_save_dir,\
                                    preload_dir = preload_dir,\
                                    linear_finetune = FLAGS.psec_linear_finetune,\
                                    nn_finetune = FLAGS.psec_nn_finetune,\
                                    opt = 'adam') 
    else:
        pi_ris = None
    
    V = None
    if FLAGS.algo == 'td':
        V = ValueFunctionWithNN(env.observation_space.shape[0], alpha, FLAGS.vf_num_hidden_layers, FLAGS.vf_num_neurons, FLAGS.vf_act_fn) 
        print ('inited tile MLP for VF')
    assert (V is not None)

    if ris:
        batch_iws = load_IWs(batch_raw, pi, pi_ris)
    else:
        # if not RIS, use actual behavior policy
        batch_iws = load_IWs(batch_raw, pi, pi_b)

    batch_vf = batch_iws
    

    true_vf = [vf[tuple(s)] for s in mc_sampled_states]
    print ('true vf {}'.format(true_vf))
   
    if FLAGS.algo == 'td':
        print ('semi gradient td')
        mse_errs = semi_gradient_n_step_td_batch(gamma, n, alpha, V, batch_vf, mc_sampled_states, true_vf = true_vf)

    Vs = [V(s)[1] for s in mc_sampled_states]
    print ('estimated vf {}'.format(Vs))
    print ('states {}'.format([s for s in mc_sampled_states]))
    
    err = mse(true_vf, Vs)
    print ('RIS: {}, MSE {}'.format(ris, err))

    results.value_error = err
    results.num_trajs = num_batch_trajs
    
    if FLAGS.outfile is not None:
        with open(FLAGS.outfile, 'wb') as w:
            w.write(results.SerializeToString())

