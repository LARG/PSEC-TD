from __future__ import print_function
from __future__ import division

import sys
import os
import argparse
import subprocess
import random
import time

parser = argparse.ArgumentParser()
parser.add_argument('result_directory', default=None, help='Directory to write results to.')
parser.add_argument('--num_trials', default=1, type=int, help='The number of trials to launch.')
parser.add_argument('--on-policy', default=False, action='store_true', help='Run on-policy experiment.')
parser.add_argument('--trajs_info', nargs='+', help='[start_num_traj, num_traj_increment, end_num_traj]')
parser.add_argument('--condor', default=False, action='store_true', help='run experiments on condor')
parser.add_argument('--predefined_batches', default=False, action='store_true', help='use predefined batches')
#parser.add_argument('--deterministic_prob', type = float, default = 1.00, help='probability agents actions work as intended')
#parser.add_argument('--alpha', type = float, default = -1, help='learning rate')
parser.add_argument('--fig_num', type = int, default = 1, help='Figure # from the paper')
FLAGS = parser.parse_args()

EXECUTABLE = "./main"

exp_args = {
    '--print_freq': 1000,
    '--behavior-number': 1
}

def gen_cmd(seed,
            outfile,
            num_trajs,
            method,
            deterministic_prob,
            lr,
            on_policy=False):
    
    """Run a single trial of a set method on Gridworld."""
    arguments = '--outfile %s --seed %d' % (outfile, seed)
    
    if method.ris:
        arguments += ' --ris'

    if method.cee_mrp:
        arguments += ' --cee-mrp'
    if method.cee_mdp:
        arguments += ' --cee-mdp'

    if method.cee_exp_sarsa_mdp:
        arguments += ' --cee-exp-sarsa-mdp'
    if method.cee_sarsa_mdp:
        arguments += ' --cee-sarsa-mdp'
    if method.cee_psec_sarsa_mdp:
        arguments += ' --cee-psec-sarsa-mdp'

    if method.mdp:
        arguments += ' --mdp'
    if method.lstd:
        arguments += ' --lstd'
    if method.rho_new_est:
        arguments += ' --rho-new-est'
    if method.state_action:
        arguments += ' --state-action '
    if method.exp_sarsa:
        arguments += ' --exp-sarsa '
   
    if on_policy:
        arguments += ' --policy-number 1'
    else:
        arguments += ' --policy-number 2'

    arguments += ' --num-trajs %s' % num_trajs
    arguments += ' --deterministic-prob %s ' % deterministic_prob
    arguments += ' --alpha %f' % lr

    for arg in exp_args:
        arguments += ' %s %s' % (arg, exp_args[arg])

    cmd = '%s %s' % (EXECUTABLE, arguments)
    return cmd

def run_trial(seed,
            outfile,
            num_trajs,
            method,
            deterministic_prob,
            lr,
            on_policy=False,
            condor = False):
    
    cmd = gen_cmd(seed,
                    outfile,
                    num_trajs,
                    method,
                    deterministic_prob,
                    lr,
                    on_policy)

    if condor:
        submitFile = 'Executable = ' + EXECUTABLE + "\n"
        submitFile += 'Error = %s.err\n' % outfile
        submitFile += 'Input = /dev/null\n' + 'Output = /dev/null\n'
        submitFile += 'Log = /dev/null\n' + "arguments = " + cmd + '\n'
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

    def __init__(self, name, cee_mrp, cee_mdp, mdp, cee_exp_sarsa_mdp, cee_sarsa_mdp, cee_psec_sarsa_mdp, ris, lstd, rho_new_est, state_action, exp_sarsa):  # noqa
        self.name = name
        self.cee_mrp = cee_mrp
        self.cee_mdp = cee_mdp
        self.cee_exp_sarsa_mdp = cee_exp_sarsa_mdp 
        self.cee_sarsa_mdp = cee_sarsa_mdp
        self.cee_psec_sarsa_mdp = cee_psec_sarsa_mdp
        
        self.mdp = mdp
        self.ris = ris
        self.lstd = lstd
        self.rho_new_est = rho_new_est
        self.state_action = state_action
        self.exp_sarsa = exp_sarsa

def main():  # noqa

    ct = 0
    if FLAGS.result_directory is None:
        print('Need to provide result directory')
        sys.exit()
    directory = FLAGS.result_directory
    seeds = [random.randint(0, 1e6) for _ in range(FLAGS.num_trials)]

    methods = []
    
    # Using a learning rate of -1 for the TD (and PSEC) methods results in
    # lr = 1. / (num_states * batch_size)
    batches = [i * 10 **exp for exp in range(0, 2) for i in range(1, 10)]

    if not FLAGS.predefined_batches:
        start_num_trajs = int(FLAGS.trajs_info[0])
        num_trajs_increment = int(FLAGS.trajs_info[1])
        end_num_trajs = int(FLAGS.trajs_info[2])
        batches = [i for i in range(start_num_trajs, end_num_trajs + 1, num_trajs_increment)] 

    # Variants of PSEC vs. TD, on- or off-policy dependent on on_policy flag
    if FLAGS.fig_num == 1:
        # learning rates
        #batches = [5,6,7,8,9,10,20]
        batches = [1,2,3,4,5,6,7,8,9,10,20, 30,40,50]#,60,70,80,90,100]
        alphas =[5e-3, 1e-3, 5e-2, 1e-2, 5e-1, -1]
        det_probs = [1.0]

        '''
        methods.append(Method('s-a-s\'-a\'',
                            cee_mrp = False,
                            cee_mdp = False,
                            mdp = True,
                            cee_exp_sarsa_mdp = False,
                            cee_sarsa_mdp = False,
                            cee_psec_sarsa_mdp = False,
                            ris = False,
                            lstd = False,
                            rho_new_est = False,
                            state_action = True,
                            exp_sarsa = False))
        '''
        methods.append(Method('SARSA(0)',
                            cee_mrp = False,
                            cee_mdp = False,
                            mdp = True,
                            cee_exp_sarsa_mdp = False,
                            cee_sarsa_mdp = False,
                            cee_psec_sarsa_mdp = False,
                            ris = False,
                            lstd = False,
                            rho_new_est = False,
                            state_action = False,
                            exp_sarsa = False))

        methods.append(Method('EXPSARSA(0)',
                            cee_mrp = False,
                            cee_mdp = False,
                            mdp = True,
                            cee_exp_sarsa_mdp = False,
                            cee_sarsa_mdp = False,
                            cee_psec_sarsa_mdp = False,
                            ris = False,
                            lstd = False,
                            rho_new_est = False,
                            state_action = True,
                            exp_sarsa = True))
        '''
        methods.append(Method('PSEC-SARSA',
                            cee_mrp = False,
                            cee_mdp = False,
                            mdp = False,
                            cee_exp_sarsa_mdp = False,
                            cee_sarsa_mdp = False,
                            cee_psec_sarsa_mdp = True,
                            ris = True,
                            lstd = False,
                            rho_new_est = False,
                            state_action = True,
                            exp_sarsa = False))
        '''

        methods.append(Method('PSEC-SARSA(0)',
                            cee_mrp = False,
                            cee_mdp = False,
                            mdp = True,
                            cee_exp_sarsa_mdp = False,
                            cee_sarsa_mdp = False,
                            cee_psec_sarsa_mdp = False,
                            ris = True,
                            lstd = False,
                            rho_new_est = True,
                            state_action = True,
                            exp_sarsa = False))
    # PSEC-lSTD and LSTD on- and off-policy depending on FLAGS
    elif FLAGS.fig_num == 3:
        # for LSTD, used "alpha" but it is epsilon (referenced in the Appendix)
        alphas =[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        det_probs = [1.0]
        methods.append(Method('LSTD',
                            cee_mrp = False,
                            cee_mdp = False,
                            mdp = True,
                            ris = False,
                            lstd = True,
                            rho_new_est = False))
        methods.append(Method('PSEC-LSTD',
                            cee_mrp = False,
                            cee_mdp = False,
                            mdp = True,
                            ris = True,
                            lstd = True,
                            rho_new_est = False))
    
    # certainty-equivalence graph
    elif FLAGS.fig_num == 41:
        alphas = [-1]
        det_probs = [1.0] # this can be any value between 0 to 1
        methods.append(Method('PSEC-TD',
                            cee_mrp = False,
                            cee_mdp = True,
                            mdp = False,
                            ris = True,
                            lstd = False,
                            rho_new_est = False))
        methods.append(Method('PSEC-TD-Estimate',
                            cee_mrp = False,
                            cee_mdp = True,
                            mdp = False,
                            ris = True,
                            lstd = False,
                            rho_new_est = True))

    # unvisited (s,a) ratio
    elif FLAGS.fig_num == 42:
        alphas = [-1]
        det_probs = [1.0] # this can be any value between 0 to 1
        # we have specified TD here, but any method works since we are just looking at
        # data, not the performance of an algorithm
        methods.append(Method('TD',
                            cee_mrp = False,
                            cee_mdp = False,
                            mdp = True,
                            ris = False,
                            lstd = False,
                            rho_new_est = False))
    # stochastcity plot
    elif FLAGS.fig_num == 43:
        alphas = [-1]
        det_probs = [i / 10. for i in range(0, 11)]
        batches = [15]
        methods.append(Method('TD(0)',
                            cee_mrp = False,
                            cee_mdp = False,
                            mdp = True,
                            ris = False,
                            lstd = False,
                            rho_new_est = False))
        methods.append(Method('PSEC-TD',
                            cee_mrp = False,
                            cee_mdp = False,
                            mdp = True,
                            ris = True,
                            lstd = False,
                            rho_new_est = False))
        methods.append(Method('PSEC-TD-Estimate',
                            cee_mrp = False,
                            cee_mdp = False,
                            mdp = True,
                            ris = True,
                            lstd = False,
                            rho_new_est = True))

    for m in batches:
        for seed in seeds:
            for method in methods:
                #if 'EXPSARSA(0)' == method.name:
                #    alphas = [ 0.01]
                #elif 'PSEC' in method.name:
                #    alphas = [ 0.005]
                #else:
                #    alphas = [0.001]
                for lr in alphas:
                    for dp in det_probs:
                        filename = os.path.join(directory, 'method_%s_trial_%d_trajs_%d_deterministic_%.1f_alpha_%f' % (method.name, seed, m, dp, lr))
                        if os.path.exists(filename):
                            continue
                        run_trial(seed,
                                filename,
                                num_trajs = m,
                                method = method,
                                deterministic_prob = dp,
                                on_policy=FLAGS.on_policy,
                                condor = FLAGS.condor,
                                lr = lr)
                        ct += 1
                        print ('submitted job number: %d' % ct)
    print('%d experiments ran.' % ct)

if __name__ == '__main__':
    main()

