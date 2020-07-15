import numpy as np
import gym
from utils import generate_batch, sample_states, mc_rollouts, mse, load_IWs 
from vf_nn import ValueFunctionWithNN
from algo import semi_gradient_n_step_td_batch
import argparse
from protos import results_pb2
import tensorflow as tf
from pdb import set_trace
from policies import GaussianPolicy

parser = argparse.ArgumentParser()

parser.add_argument('--env_name', type = str, help='saving result file', default = None)

parser.add_argument('--outfile', type = str, help='saving result file', default = None)
parser.add_argument('--seed', type = int, help='seed for randomness')

parser.add_argument('--algo', type = str, default = 'td', help='td')
parser.add_argument('--ris', default = False, action='store_true', help='if applying RIS')
parser.add_argument('--num_trajs', default = 50, type=int, help='num trajs for batch')

# evaluation/behavior policy to execute
parser.add_argument('--pi', default = 0, type=int, help='0 for StochasticBangBang, 1 for Linear, 2 for neural net')
parser.add_argument('--pi_b', default = 0, type=int, help='0 for StochasticBangBang, 1 for Linear, 2 for neural net')
parser.add_argument('--pi_ris', default = 0, type=int, help='0 for StochasticBangBang approx, 1 for Linear, 2 for neural net')
parser.add_argument('--pi_weightfile', default = str, help='weight file for behavior policy')
parser.add_argument('--pi_b_weightfile', default = str, help='weight file for behavior policy')
parser.add_argument('--pi_b_psec_weightfile', type=str, default = None, help='weightfile of behavior policy')

# NN PSEC specs
parser.add_argument('--psec_alpha', default = 0, type=float, help='learning rate')
parser.add_argument('--psec_num_neurons', default = 16, type=int, help='number of tilings for VF')
parser.add_argument('--psec_num_hidden_layers', default = 2, type=int, help='number of tilings for VF')
parser.add_argument('--psec_act_fn', default = 0, type=int, help='number of tilings for VF')
parser.add_argument('--psec_linear_finetune', default = False, action='store_true', help='if applying RIS')
parser.add_argument('--psec_nn_finetune', default = False, action='store_true', help='if applying RIS')

# MC rollout related for true VF comparison
parser.add_argument('--num_mc_trajs', default = 300, type=int, help='The number of trajs to collect before sampling MC states.')
parser.add_argument('--num_mc_sampled_states', default = 200, type=int, help='The number of states to sampled for MC rollouts.')
parser.add_argument('--num_mc_rollouts', default = 300, type=int, help='The number of MC rollouts to determine "true" VF')

# MDP specifics for training
parser.add_argument('--alpha', default = 0, type=float, help='learning rate')
parser.add_argument('--gamma', default = 1, type=float, help='discount factor')
parser.add_argument('--n', default = 1, type=int, help='n-step (RIS) TD')

# vf function specifics
parser.add_argument('--vf_rep', type = str, default = 'mlp', help='function approximator for value function')
parser.add_argument('--vf_num_neurons', default = 256, type=int, help='number of tilings for VF')
parser.add_argument('--vf_num_hidden_layers', default = 1, type=int, help='number of tilings for VF')
parser.add_argument('--vf_act_fn', default = 0, type=int, help='number of tilings for VF')


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
    if FLAGS.pi == 4:
        pi = GaussianPolicy(env.observation_space, env.action_space, scope='pi', train_type = 'reinforce', hidden_sizes = [64, 64], act_fn = tf.nn.tanh)
        #pi = GaussianPolicy(env.observation_space, env.action_space, scope='pi', train_type = 'reinforce', hidden_sizes = [16, 16])
        pi.load_policy(FLAGS.pi_weightfile)

    # setting behavior policy
    if FLAGS.pi_b == 4:
        pi_b = pi
 
    num_mc_trajs = FLAGS.num_mc_trajs
    num_mc_sampled_states = FLAGS.num_mc_sampled_states
    num_mc_rollouts = FLAGS.num_mc_rollouts

    # evaluating "true" value of certain states based on evaluation policy
    # generate trajectories to sample states
    mc_trajs = generate_batch(env, env_name, pi, num_mc_trajs)
    # sample states to compute MC estimates for
    mc_sampled_states = sample_states(mc_trajs, num_mc_sampled_states)
    assert len(mc_sampled_states) == num_mc_sampled_states
    #assert len(mc_sampled_states[0]) == env.observation_space.shape[0]
    assert len(mc_sampled_states[0][0]) == env.observation_space.shape[0]
    # get MC estimates for sampled states (as "true" value)
    vf = mc_rollouts(env, env_name, pi, num_mc_rollouts, mc_sampled_states)

    # removing the full state
    mc_sampled_states = [s[0] for s in mc_sampled_states]

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
        val_batch_raw = generate_batch(env, env_name, pi_b, max(int(0.2 * num_batch_trajs), 1))
        psec_save_dir = None
        if FLAGS.outfile is not None:
            psec_save_dir = FLAGS.outfile.split('/')[2]
        # used ONLY for tc not for any approximation
        if FLAGS.pi_ris == 4:
            print ('RIS NN policy Gaussian policy')
            hidden_sizes = [FLAGS.psec_num_neurons for _ in range(FLAGS.psec_num_hidden_layers)]
            def lrelu(x):
                return tf.nn.leaky_relu(x, alpha = 0.01)
            pi_ris = GaussianPolicy(env.observation_space,\
                                    env.action_space,\
                                    scope='mle',\
                                    hidden_sizes = hidden_sizes,\
                                    save_dir = psec_save_dir,\
                                    learning_rate = FLAGS.psec_alpha,
                                    linear_finetune = FLAGS.psec_linear_finetune, act_fn = tf.nn.tanh)

            print ('finetune linear {}, NN {}'.format(FLAGS.psec_linear_finetune, FLAGS.psec_nn_finetune))
            if FLAGS.psec_linear_finetune or FLAGS.psec_nn_finetune:
                print ('loaded policy to tune')
                pi_ris.load_policy(FLAGS.pi_b_psec_weightfile)
            pi_ris.compute_psec(batch_raw, val_batch_raw)

    else:
        pi_ris = None
    
    # width based on getting 10 tiles in each dimension (acc to White paper)

    V = None
    if FLAGS.vf_rep == 'mlp':
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
   
    semi_gradient_n_step_td_batch(gamma, n, alpha, V, batch_vf, mc_sampled_states, true_vf = true_vf)

    Vs = [V(s)[1] for s in mc_sampled_states]
    print ('estimated vf {}'.format(Vs))
    
    err = mse(true_vf, Vs)
    print ('RIS: {}, MSE {}'.format(ris, err))

    results.value_error = err
    results.num_trajs = num_batch_trajs
    
    if FLAGS.outfile is not None:
        with open(FLAGS.outfile, 'wb') as w:
            w.write(results.SerializeToString())

