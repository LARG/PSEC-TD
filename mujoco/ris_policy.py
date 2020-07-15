import numpy as np
from math import ceil
from policy_struct import LinearPolicy, NN
import utils
import time
from pdb import set_trace
import os, shutil

class RISCountBasedPolicy():
    def __init__(self, env, batch:np.array, V):
        self.env = env
        self.batch = batch
        self.V = V
        self.s_a_counts = {}
        self.s_counts = {}
        self._compute_ris_estimates()

    def _compute_ris_estimates(self):
       
        batch = self.batch 
        for traj in batch:
            for idx, step in enumerate(traj):
                s = step[0]
                a = step[1]

                feat = s
                if self.V is not None: 
                    feat = self.V.get_feature_vec(s)
                feat = tuple(feat)

                if feat not in self.s_counts:
                    self.s_counts[feat] = 0

                if feat not in self.s_a_counts:
                    self.s_a_counts[feat] = {}

                if a not in self.s_a_counts[feat]:
                    self.s_a_counts[feat][a] = 0

                self.s_counts[feat] += 1
                self.s_a_counts[feat][a] += 1

        #assert abs(sum(self.action_probs[0]) - 1.) <= 1e-2
        #assert abs(sum(self.action_probs[1]) - 1.) <= 1e-2

    def action_prob(self, state, act):

        feat = state
        if self.V is not None:
            feat = self.V.get_feature_vec(state)#[0]
        feat = tuple(feat)
        return self.s_a_counts[feat][act] / self.s_counts[feat]


class RISRandomActionPolicy():

    def __init__(self, env, batch:np.array):
        self.env = env
        self.batch = batch
        self.action_probs = np.zeros((2, env.action_space.n))
        self._compute_ris_estimates()

    def _compute_ris_estimates(self):
       
        neg_total_actions = 0
        pos_total_actions = 0
        batch = self.batch 
        for traj in batch:
            for idx, step in enumerate(traj):
                s = step[0][1] # velocity
                a = step[1]
                if s < 0:
                    self.action_probs[0, a] += 1
                    neg_total_actions += 1
                else:
                    self.action_probs[1, a] += 1
                    pos_total_actions += 1
        self.action_probs[0] /= neg_total_actions
        self.action_probs[1] /= pos_total_actions
        assert abs(sum(self.action_probs[0]) - 1.) <= 1e-2
        assert abs(sum(self.action_probs[1]) - 1.) <= 1e-2

    def action_prob(self, state, act):
        if state[1] < 0:
            return self.action_probs[0, act]
        else:
            return self.action_probs[1, act]

class RISLinearPolicy():
    def __init__(self, env, batch:np.array, V = None, lr = 1e-4, pi_eval = None):
        self.env = env
        self.batch = batch

        # for tc
        self.V = V

        if V is not None:
            self.pi = LinearPolicy(V.get_feature_vec_len(), env.action_space.n, lr)
        else:
            self.pi = LinearPolicy(env.observation_space.shape[0], env.action_space.n, lr)
        self.pi_eval = pi_eval

        # processing batch of data, generating 1-hot vectors
        data_x, data_y = utils.pre_process_batch(env, None, batch, V)

        self.train_x, self.train_y, _, _ = utils.split_data(data_x, data_y, ratio = 0.0)
        assert len(self.train_x) == len(data_x)

        # generating separate validation set
        #val_batch = utils.generate_batch(self.env, self.pi_eval, 0.05 * len(self.batch))
        #self.test_x, self.test_y = utils.pre_process_batch(env, None, val_batch, V)
        print ('Number of training trajs {}'.format(len(self.batch))) 
        print ('Number of training steps {}'.format(len(self.train_x))) 
        #print ('Number of testing trajs {}'.format(len(val_batch))) 
        #print ('Number of testing steps {}'.format(len(self.test_x))) 

        self.compute_ris_estimates()

    def action(self,s):
        f = s
        if self.V is not None:
            f = self.V.get_feature_vec(s)#[0]
        features = np.reshape(f, (1, len(f)))
        return self.pi.action(features)

    def action_prob(self, s, a):
        f = s
        if self.V is not None:
            f = self.V.get_feature_vec(s)#[0]
        features = np.reshape(f, (1, len(f)))
        return self.pi.action_prob(features, a)

    def compute_ris_estimates(self):

        patience_count = 0
        patience_limit = 20
        print_freq = 100
        prev_val_loss = float('inf')
        prev_tr_loss = float('inf')
        epochs = 500000

        for e in range(epochs):
            self.pi.train(self.train_x, self.train_y)
            #self.pi.validate(self.test_x, self.test_y)

            tr_loss = self.pi.tr_loss
            #val_loss = self.pi.val_loss

            if e % print_freq == 0:
                #print ('epoch {}, training loss {}, validation loss {}'.format(e, tr_loss, val_loss))
                print ('epoch {}, training loss {}'.format(e, tr_loss))#, val_loss))

            if abs(tr_loss - prev_tr_loss) <= 1e-7:
                patience_count += 1
            prev_tr_loss = tr_loss

            if patience_count >= patience_limit:
                break
            break
            '''
            if val_loss > prev_val_loss:
                patience_count += 1
            if patience_count >= patience_limit:
                break
            prev_val_loss = val_loss
            '''
class RISNNPolicy():
    def __init__(self, env, tr_batch:np.array, val_batch, num_hidden_layers,
                    num_neurons,
                    act_fn,
                    lr,
                    comp_ris = True,
                    pi_eval = None,
                    save_dir = None,
                    preload_dir = None,
                    linear_finetune = False,
                    nn_finetune = False,
                    opt = 'gd'):
        self.env = env
        self.tr_batch = tr_batch
        self.val_batch = val_batch

        if preload_dir is not None:
            ckpt_split = preload_dir.split('/')
            ckpt = ('/'.join(ckpt_split + [ckpt_split[1]])) + '-0'
            preload_dir = ckpt

        self.pi = NN(env.observation_space.shape[0],
                        env.action_space.n,
                        num_hidden_layers,
                        num_neurons,
                        act_fn,
                        lr,
                        ckpt = preload_dir,
                        linear_finetune = linear_finetune,
                        nn_finetune = nn_finetune,
                        scope_name = 'finetunepgpol',
                        opt = opt)
        self.pi_eval = pi_eval

        # processing batch of data, generating 1-hot vectors
        data_x, data_y = utils.pre_process_batch(env, None, tr_batch, V = None)

        self.train_x, self.train_y, _, _ = utils.split_data(data_x, data_y, ratio = 0.0)
        assert len(self.train_x) == len(data_x)
        # generating separate validation set
        data_x, data_y = utils.pre_process_batch(env, None, val_batch, V = None)
        self.val_x, self.val_y, _, _ = utils.split_data(data_x, data_y, ratio = 0.0)

        #val_batch = utils.generate_batch(self.env, self.pi_eval, 0.05 * len(self.batch))
        #self.test_x, self.test_y = utils.pre_process_batch(env, None, val_batch, V)
        print ('Number of training trajs {}'.format(len(self.tr_batch))) 
        print ('Number of training steps {}'.format(len(self.train_x))) 
        print ('Number of testing trajs {}'.format(len(val_batch))) 
        print ('Number of testing steps {}'.format(len(self.val_x)))


        # saving setup for early stopping
        self.psec_pi_dir = 'psec_pi_dir/'
        if save_dir is not None:
            self.psec_pi_dir = save_dir
        print ('saving psec net in {}'.format(self.psec_pi_dir))
        if os.path.isdir(self.psec_pi_dir):
            shutil.rmtree(self.psec_pi_dir)
        os.mkdir(self.psec_pi_dir) 
        clct_ckpt_dir = os.path.join(self.psec_pi_dir, 'pi')
        self.clct_ckpt_file = os.path.join(clct_ckpt_dir, 'pi') 

        if comp_ris:
            self.compute_ris_estimates()

    def action(self,s):
        s = np.reshape(s, (1, len(s)))
        return self.pi.action(s)

    def action_prob(self, s, a):
        s = np.reshape(s, (1, len(s)))
        return self.pi.action_prob(s, a)

    def step_train(self):
        self.pi.train(self.train_x, self.train_y)
        self.pi.validate(self.val_x, self.val_y)
        tr_loss = self.pi.tr_loss
        val_loss = self.pi.val_loss

        return tr_loss, val_loss 

    def compute_ris_estimates(self):

        patience_count = 0
        patience_limit = 15
        print_freq = 100
        prev_val_loss = float('inf')
        prev_tr_loss = float('inf')
        epochs = 100#20000#500000
        min_val_loss = float('inf')
        min_val_itr = 0

        for e in range(epochs):
            s = time.time()
            #inds = np.random.choice(self.train_x.shape[0], size=1)#self.train_x.shape[0])
            #train_x = self.train_x[inds,:]
            #train_y = self.train_y[inds,:]
            #self.pi.train(train_x, train_y)
            self.pi.train(self.train_x, self.train_y)
            #assert len(self.train_x) == len(train_x)
            l = time.time()
            #print ('training took {}'.format(l - s))
            #inds = np.random.choice(self.val_x.shape[0], size=self.val_x.shape[0])
            #self.pi.validate(self.val_x[inds, :], self.val_y[inds, :])
            self.pi.validate(self.val_x, self.val_y)

            tr_loss = self.pi.tr_loss
            val_loss = self.pi.val_loss

            if e % print_freq == 0:
                print ('epoch {}, training loss {}, validation loss {}'.format(e, tr_loss, val_loss))
                #print ('epoch {}, training loss {}'.format(e, tr_loss))#, val_loss))

            # checking slope of validation error
            '''
            error_deg = abs(np.rad2deg(np.arctan2(val_loss - prev_val_loss, 1)))
            if error_deg <= 30:
                break
            prev_val_loss = val_loss
            '''


            # overfitting critera
            '''
            if abs(tr_loss - prev_tr_loss) <= 1e-5:
                patience_count += 1
            else:
                # reset
                patience_count = 0
            prev_tr_loss = tr_loss

            if patience_count >= patience_limit:
                break
            '''

            if val_loss < min_val_loss:
                # error improved!
                min_val_loss = val_loss
                min_val_itr = e
                self.pi.save(self.clct_ckpt_file, min_val_itr)
                #print ('got min at iteration {}'.format(min_val_itr))

            # if its been at least 100 iterations since we last recorded a min, then break
            if (e - min_val_itr) >= 50:
                break

        min_ckpt_file = self.clct_ckpt_file+'-'+str(min_val_itr)
        self.pi.load(min_ckpt_file)
        shutil.rmtree(self.psec_pi_dir)

