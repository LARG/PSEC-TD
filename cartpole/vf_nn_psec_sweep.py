from __future__ import print_function
from __future__ import division

import sys
import os
import argparse
import subprocess
import random
import time

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type = str, default = None, help='function approximator for value function')
parser.add_argument('result_directory', default = None, help='Directory to write results to.')
parser.add_argument('--num_trials', default = 1, type=int, help='The number of trials to launch.')
parser.add_argument('--trajs_info', nargs = '+', help='[start_num_traj, num_traj_increment, end_num_traj]')
parser.add_argument('--condor', default = False, action='store_true', help='run experiments on condor')

# evaluation/behavior policy to execute
parser.add_argument('--pi', default = 0, type=int, help='0 for StochasticBangBang, 1 for Linear, 2 for neural net')
parser.add_argument('--pi_b', default = 0, type=int, help='0 for StochasticBangBang, 1 for Linear, 2 for neural net')
parser.add_argument('--on_policy', default = False, action='store_true', help='run experiments on condor')
parser.add_argument('--pi_ris', default = 0, type=int, help='0 for StochasticBangBang, 1 for Linear, 2 for neural net')
parser.add_argument('--pi_weightfile', type=str, default = None, help='weightfile of behavior policy')
parser.add_argument('--pi_b_weightfile', type=str, default = None, help='weightfile of behavior policy')
parser.add_argument('--pi_b_psec_weightfile', type=str, default = None, help='weightfile of behavior policy')

# MC rollout related for true VF comparison
parser.add_argument('--num_mc_trajs', default = 300, type=int, help='The number of trajs to collect before sampling MC states.')
parser.add_argument('--num_mc_sampled_states', default = 200, type=int, help='The number of states to sampled for MC rollouts.')
parser.add_argument('--num_mc_rollouts', default = 300, type=int, help='The number of MC rollouts to determine "true" VF')

# MDP specifics for training
parser.add_argument('--gamma', default = 1, type=float, help='discount factor')
parser.add_argument('--n', default = 1, type=int, help='n-step (RIS) TD')

# linear function specifics
parser.add_argument('--vf_rep', type = str, default = 'linear', help='function approximator for value function')
parser.add_argument('--num_tilings', default = 10, type=int, help='number of tilings for VF')
parser.add_argument('--num_tiles', default = 10, type=int, help='number of tiles in each dimension (warning: constant for all dimensions).')

parser.add_argument('--loss_grad_exp', default = False, action='store_true', help='run experiments on condor')

FLAGS = parser.parse_args()

#EXECUTABLE = 'experiments.py'
EXECUTABLE = 'exp.sh'
if FLAGS.loss_grad_exp:
    EXECUTABLE = 'exp_loss_grad.sh'

def gen_cmd(seed,
            outfile,
            num_trajs,
            alpha,
            method,
            vf_act,
            vf_nh,
            vf_ne,
            psec_act,
            psec_nh,
            psec_ne,
            psec_lr):
    
    """Run a single trial of a set method on Gridworld."""
    arguments = '--env_name %s --outfile %s --seed %d' % (FLAGS.env_name, outfile, seed)
    
    if method.ris:
        arguments += ' --ris'
        arguments += ' --pi_ris %d' % FLAGS.pi_ris
        arguments += ' --psec_act_fn %d ' % psec_act 
        arguments += ' --psec_num_hidden_layers %d ' % psec_nh
        arguments += ' --psec_num_neurons %d ' % psec_ne 
        arguments += ' --psec_alpha %f ' % psec_lr
        if method.psec_linear_finetune:
            arguments += ' --psec_linear_finetune '
        elif method.psec_nn_finetune:
            arguments += ' --psec_nn_finetune ' 

    arguments += ' --num_trajs %s' % num_trajs
    
    arguments += ' --pi %d' % FLAGS.pi

    if FLAGS.on_policy:
        arguments += ' --pi_b %d' % FLAGS.pi
    else:
        arguments += ' --pi_b %d' % FLAGS.pi_b

    arguments += ' --pi_weightfile %s' % FLAGS.pi_weightfile
    arguments += ' --pi_b_weightfile %s' % FLAGS.pi_b_weightfile
    arguments += ' --pi_b_psec_weightfile %s' % FLAGS.pi_b_psec_weightfile
    
    arguments += ' --num_mc_trajs %d' % FLAGS.num_mc_trajs
    arguments += ' --num_mc_sampled_states %d' % FLAGS.num_mc_sampled_states
    arguments += ' --num_mc_rollouts %d' % FLAGS.num_mc_rollouts
    
    arguments += ' --gamma %f' % FLAGS.gamma
    arguments += ' --n %d' % FLAGS.n
    arguments += ' --alpha %f' % alpha
   
    arguments += ' --vf_rep %s ' % FLAGS.vf_rep 
    arguments += ' --vf_act_fn %d' % vf_act
    arguments += ' --vf_num_hidden_layers %d' % vf_nh
    arguments += ' --vf_num_neurons %d' % vf_ne

    if 'EXP-SARSA' in method.name:
        arguments += ' --algo expsarsa '
    elif 'SARSA' in method.name:
        arguments += ' --algo sarsa '

    if FLAGS.condor:
        cmd = '%s' % (arguments)
    else:
        EXECUTABLE = 'experiments.py'
        if FLAGS.loss_grad_exp:
            EXECUTABLE = 'loss_vs_mse.py'
        cmd = 'python3 %s %s' % (EXECUTABLE, arguments)
    return cmd

def run_trial(seed,
            outfile,
            num_trajs,
            method,
            alpha,
            vf_act,
            vf_nh,
            vf_ne,
            psec_act = None,
            psec_nh = None,
            psec_ne = None,
            psec_lr = None,
            condor = False):
   
    cmd = gen_cmd(seed,
                    outfile,
                    num_trajs,
                    alpha,
                    method,
                    vf_act,
                    vf_nh,
                    vf_ne,
                    psec_act,
                    psec_nh,
                    psec_ne,
                    psec_lr)

    if condor:
        submitFile = 'Executable = ' + EXECUTABLE + "\n"
        submitFile += 'Error = %s.err\n' % outfile
        submitFile += 'Input = /dev/null\n' + 'Output = /dev/null\n'
        submitFile += 'Log = /dev/null\n' + "arguments = " + cmd + '\n'
        #submitFile += 'Log = %s.log\n' % outfile + "arguments = " + cmd + '\n'
        submitFile += 'Universe = vanilla\n'
        submitFile += 'Getenv = true\n'
        submitFile += 'Requirements = ARCH == "X86_64" && !GPU\n'
        submitFile += 'Rank = -SlotId + !InMastodon*10\n'
        submitFile += '+Group = "GRAD"\n' + '+Project = "AI_ROBOTICS"\n'
        submitFile += '+ProjectDescription = "TD(0) vs RIS-TD(0)"\n'
        submitFile += 'Queue'

        proc = subprocess.Popen('condor_submit', stdin=subprocess.PIPE)
        proc.stdin.write(submitFile.encode())
        proc.stdin.close()
        time.sleep(0.3)
    else:
        subprocess.Popen(cmd.split())

class Method(object):
    """Object for holding method params."""

    def __init__(self, name, ris, lstd, psec_linear_finetune = False, psec_nn_finetune = False):  # noqa
        self.name = name
        self.ris = ris
        self.lstd = lstd
        self.psec_linear_finetune = psec_linear_finetune
        self.psec_nn_finetune = psec_nn_finetune
        print ('name {} psec linear {} psec nn {}'.format(name, self.psec_linear_finetune, self.psec_nn_finetune))

def main():  # noqa
    ct = 0
    if FLAGS.result_directory is None:
        print('Need to provide result directory')
        sys.exit()
    directory = FLAGS.result_directory
    seeds = [random.randint(0, 1e6) for _ in range(FLAGS.num_trials)]

    # lr goes up to -14
    vf_act_fn = [0]
    #vf_num_hidden_layers = [1, 0] # HACKY BUT IMPORTANT, check if statement in loops
    vf_num_hidden_layers = [1]#[1]#[4] # HACKY BUT IMPORTANT, check if statement in loops
    vf_num_neurons = [512]#[128]
   
    psec_act_fns = [0]
    #psec_num_hidden_layers = [2, 3, 4]
    #psec_num_hidden_layers = [1,2,3]
    psec_num_hidden_layers = [3]
    psec_num_neurons = [16]#[16, 32]
    methods = []
   
    methods.append(Method('TD(0)', ris = False, lstd = False))
    methods.append(Method('PSEC-TD(0)', ris = True, lstd = False))
    #methods.append(Method('Lin-Finetune-PSEC-TD(0)', ris = True, lstd = False, psec_linear_finetune = True))
    #methods.append(Method('NN-Finetune-PSEC-TD(0)', ris = True, lstd = False, psec_nn_finetune = True))
    
    start_num_trajs = int(FLAGS.trajs_info[0])
    num_trajs_increment = int(FLAGS.trajs_info[1])
    end_num_trajs = int(FLAGS.trajs_info[2])
  

    #trajs = [1, 5, 10, 50, 100, 500]
    trajs = [10, 50, 100, 500, 1000]
    for m in range(start_num_trajs, end_num_trajs + 1, num_trajs_incremen:t):
        for vf_act in vf_act_fn:
            vf_alphas = [1.0]
            for a in vf_alphas:
                for vf_nh in vf_num_hidden_layers:
                    if vf_nh == 0:
                        vf_num_neurons = [0]
                    for vf_ne in vf_num_neurons:
                        for seed in seeds:
                            for method in methods:
                                if 'PSEC' in method.name:
                                    psec_alphas = [0.1 * 2 ** i for i in range(-6,-1)]
                                    for psec_act in psec_act_fns:
                                        for psec_nh in psec_num_hidden_layers:
                                            for psec_ne in psec_num_neurons:
                                                for psec_lr in psec_alphas:
                                                    filename = os.path.join(directory, \
                                                        'method_%s_trial_%d_trajs_%d_alpha_%f_vf-act_%d_vf-nh_%d_vf-ne_%d_psec-act_%d_psec-nh_%d_psec-ne_%d_psec-lr_%f'\
                                                            % (method.name, seed, m, a, vf_act, vf_nh, vf_ne, psec_act, psec_nh, psec_ne, psec_lr))
                                                    if os.path.exists(filename):
                                                        continue
                                                    run_trial(seed,
                                                            filename,
                                                            num_trajs = m,
                                                            method = method,
                                                            alpha = a,
                                                            condor = FLAGS.condor,
                                                            vf_act = vf_act,
                                                            vf_nh = vf_nh,
                                                            vf_ne = vf_ne,
                                                            psec_act = psec_act,
                                                            psec_nh = psec_nh,
                                                            psec_ne = psec_ne,
                                                            psec_lr = psec_lr)
                                                    ct += 1
                                                    print ('submitted job number: %d' % ct)
                                else:
                                    filename = os.path.join(directory, \
                                        'method_%s_trial_%d_trajs_%d_alpha_%f_vf-act_%d_vf-nh_%d_vf-ne_%d_psec-act_%d_psec-nh_%d_psec-ne_%d_psec-lr_%f'\
                                            % (method.name, seed, m, a, vf_act, vf_nh, vf_ne, 0, 0, 0, 0))
                                    if os.path.exists(filename):
                                        continue
                                    run_trial(seed,
                                            filename,
                                            num_trajs = m,
                                            method = method,
                                            alpha = a,
                                            condor = FLAGS.condor,
                                            vf_act = vf_act,
                                            vf_nh = vf_nh,
                                            vf_ne = vf_ne)
                                    ct += 1
                                    print ('submitted job number: %d' % ct)
    print('%d experiments ran.' % ct)

    print (FLAGS.trajs_info) 
    print (seeds)

if __name__ == "__main__":
    main()


