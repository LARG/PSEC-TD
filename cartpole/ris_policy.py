import numpy as np
from math import ceil
from policy_struct import NN
import utils
import time
from pdb import set_trace
import os, shutil

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
        epochs = 20000 
        min_val_loss = float('inf')
        min_val_itr = 0
        
        for e in range(epochs):
            s = time.time()
            self.pi.train(self.train_x, self.train_y)
            l = time.time()
            self.pi.validate(self.val_x, self.val_y)

            tr_loss = self.pi.tr_loss
            val_loss = self.pi.val_loss

            if e % print_freq == 0:
                print ('epoch {}, training loss {}, validation loss {}, best itr so far {}'.format(e, tr_loss, val_loss, min_val_itr))
                #print ('epoch {}, training loss {}'.format(e, tr_loss))#, val_loss))

            if val_loss < min_val_loss:
                # error improved!
                min_val_loss = val_loss
                min_val_itr = e
                self.pi.save(self.clct_ckpt_file, min_val_itr)
                #print ('got min at iteration {}'.format(min_val_itr))

            # if its been at least 500 iterations since we last recorded a min, then break
            if (e - min_val_itr) >= 500:
                break

        min_ckpt_file = self.clct_ckpt_file+'-'+str(min_val_itr)
        self.pi.load(min_ckpt_file)
        shutil.rmtree(self.psec_pi_dir)

