import numpy as np
import tensorflow as tf
from pdb import set_trace
import os
import shutil

def _get_space_size(space):
    if hasattr(space, 'n'):
        return space.n
    elif hasattr(space, 'low'):
        return np.size(space.low, axis=0)


def _get_space_dims(space):
    if hasattr(space, 'n'):
        return 1
    elif hasattr(space, 'low'):
        return space.low.shape


def get_policy_class(policy_str):
    if policy_str == 'boltzmann':
        return ContinuousStateBoltzmannPolicy
    if policy_str in ['gaussian', 'Gaussian']:
        return GaussianPolicy


class Policy(object):

    def __init__(self, observation_space, action_space):
        pass

    def get_action(self, observation):
        pass

    def pdf(self, observation, action):
        pass

    def grad_log_policy(self, observation, action):
        pass


class RandomPolicy(Policy):

    def __init__(self, observation_space, action_space):
        self.action_space = action_space
        #if isinstance(self.action_space, spaces.Box):
        #    self.r = self.action_space.high - self.action_space.low
        #else:
        #    self.r = self.action_space.n

    def get_action(self, observation):
        return self.action_space.sample(), 1. / self.r

    def pdf(self, observation, action):
        return 1. / self.r

    def grad_log_policy(self, obs, act):
        return 0.


class NeuralNetworkPolicy(Policy):
    """
    Policy that computes action selection distribution with a neural net
    function of the given observation. Base class for GaussianPolicy and
    ContinuousStateBoltzmannPolicy.
    """

    def __init__(self, observation_space, action_space, scope=None,
                 learning_rate=1e-04,
                 hidden_sizes=[],
                 act_fn=tf.nn.relu,
                 filter_obs=False,
                 seed=None,
                 entropy_coeff=None,
                 weight_decay=0.0,
                linear_finetune = False):
        """
        observation_space:
        n_logits:
        scope:
        learning_rate:
        hidden_sizes:
        train_type:
        act_fn:
        filter_obs:
        seed:
        entropy_coeff:
        """
        obs_dim = _get_space_size(observation_space)
        n_logits = _get_space_size(action_space)
        #self.graph = tf.Graph()
        scope = '' if scope is None else scope
        self.scope = scope
        self.entropy_coeff = entropy_coeff
        self.learning_rate = learning_rate
        if seed is not None:
            np.random.seed(seed)

        self.session = tf.Session()
        #with self.graph.as_default():

        with tf.variable_scope(scope):

            self.params = []
            self.obs_input = tf.placeholder(tf.float32, shape = [None, obs_dim])#, name = 'obs_input')
            #self.obs_input = tflearn.input_data(shape=[None, obs_dim],
            #                                    name='obs_input')
            self.obs_dim = tuple([-1, obs_dim])

            if filter_obs:
                with tf.variable_scope('obfilter'):

                    self.rms_sum = tf.get_variable(
                        dtype=tf.float64,
                        shape=obs_dim,
                        initializer=tf.constant_initializer(1.0),
                        name='runningsum', trainable=False)
                    self.rms_sumsq = tf.get_variable(
                        dtype=tf.float64,
                        shape=obs_dim,
                        initializer=tf.constant_initializer(1.0),
                        name='runningsumsq', trainable=False)
                    self.rms_count = tf.get_variable(
                        dtype=tf.float64,
                        shape=(),
                        initializer=tf.constant_initializer(1.0),
                        name='count', trainable=False)
                    mean = tf.to_float(self.rms_sum / self.rms_count)
                    var = tf.to_float(self.rms_sumsq / self.rms_count)
                    var = var - tf.square(mean)
                    var = tf.maximum(var, 1e-2)
                    std = tf.sqrt(var)
                    self.params.extend([self.rms_sum, self.rms_sumsq,
                                        self.rms_count])

                prev = tf.clip_by_value((self.obs_input - mean) / std,
                                        -5.0, 5.0)
            else:
                prev = self.obs_input


            trainable_hidden_w = True
            trainable_linear_w = True
            if linear_finetune:
                trainable_hidden_w = False
            init = tf.contrib.layers.xavier_initializer()
            #init = tf.contrib.layers.variance_scaling_initializer()
            #init = tf.truncated_normal_initializer()
            #init = tflearn.initializations.truncated_normal(seed=seed)
            print ('training hidden {}, linear {}'.format(trainable_hidden_w, trainable_linear_w))
            for idx, size in enumerate(hidden_sizes):
                #prev = tf.layers.dense(prev, size, kernel_initializer = init, name = 'fc%i'%(idx+1), activation = act_fn)
                #, 
                prev = tf.layers.dense(prev, size,\
                        kernel_initializer = init,\
                        activation = act_fn, name = "fc%i"%(idx+1),\
                        kernel_regularizer = tf.contrib.layers.l2_regularizer(weight_decay), trainable = trainable_hidden_w)
                self.params.extend([prev])
            #self.logits = tf.squeeze(tf.layers.dense(prev, n_logits, activation = None, kernel_initializer= init, name = 'final'))
            #self.logits = tf.layers.dense(prev, n_logits, activation = None, kernel_initializer= init, name = 'final')
            #name = 'final'
            self.logits = tf.layers.dense(prev, n_logits, activation = None, \
                kernel_initializer= init, name = 'final',\
                kernel_regularizer = tf.contrib.layers.l2_regularizer(weight_decay), trainable = trainable_linear_w)
            self.params.extend([self.logits])

            '''
            for idx, size in enumerate(hidden_sizes):
                prev = tflearn.fully_connected(prev, size,
                                               name='hidden_layer%d' % idx,
                                               activation=act_fn,
                                               weights_init=init,
                                               weight_decay=weight_decay)
                self.params.extend([prev.W, prev.b])

            self.logits = tflearn.fully_connected(prev, n_logits,
                                                  name='logits',
                                                  weights_init=init,
                                                  weight_decay=weight_decay)
            self.params.extend([self.logits.W, self.logits.b])
            '''

    def initialize(self):
        self.session.run(self.init_op)

    def reinforce_update(self, observations, actions, advantages):
        feed_dict = {self.obs_input: observations,
                     self.adv_var: advantages}
        _, loss = self.session.run([self.train_step, self.loss],
                                   feed_dict=feed_dict)
        return loss

    def compute_psec(self, tr_batch, val_batch):
        train_obs = np.array([s[0] for traj in tr_batch for s in traj])
        train_acs = np.array([s[1] for traj in tr_batch for s in traj])

        val_obs = np.array([s[0] for traj in val_batch for s in traj])
        val_acs = np.array([s[1] for traj in val_batch for s in traj])

        patience_count = 0
        patience_limit = 15
        print_freq = 100
        prev_val_loss = float('inf')
        prev_tr_loss = float('inf')
        epochs = 25000#500000
        min_val_loss = float('inf')
        min_val_itr = 0

        for e in range(epochs):
            #inds = np.random.choice(train_obs.shape[0], size=2048)#self.train_x.shape[0])
            #train_x = train_obs[inds,:]
            #train_y = train_acs[inds,:]
            #self.supervised_update(train_x, train_y)

            #tr_loss = self.eval_loss(train_x, train_y)
            self.supervised_update(train_obs, train_acs)
            tr_loss = self.eval_loss(train_obs, train_acs)
            val_loss = self.eval_loss(val_obs, val_acs)
            #tr_loss = self.pi.tr_loss
            #val_loss = self.pi.val_loss

            if e % print_freq == 0:
                print ('epoch {}, training loss {}, validation loss {}'.format(e, tr_loss, val_loss))

            '''
            if val_loss > prev_val_loss:
                patience_count += 1
            else:
                patience_count = 0

            if patience_count >= patience_limit:
                break

            prev_val_loss = val_loss
            '''
            if val_loss < min_val_loss:
                # error improved!
                min_val_loss = val_loss
                min_val_itr = e
                self.save_policy(self.clct_ckpt_file, min_val_itr)
                #self.pi.save(self.clct_ckpt_file, min_val_itr)
                #self.saver.save(self.sess, dir_path, global_step=num_iter)
                #print ('got min at iteration {}'.format(min_val_itr))

            # if its been at least 100 iterations since we last recorded a min, then break
            if (e - min_val_itr) >= 500:
                break
        min_ckpt_file = self.clct_ckpt_file+'-'+str(min_val_itr)
        self.load_policy(min_ckpt_file, latest_checkpoint = False)
        shutil.rmtree(self.psec_pi_dir)

    def supervised_update(self, observations, actions):
        act_in = actions
        if hasattr(self, 'n_actions'):
            acts = np.zeros(shape=(len(actions), self.n_actions))
            for i, act in enumerate(actions):
                acts[i, act] = 1.
            act_in = acts
        feed_dict = {self.obs_input: observations,
                     self.act_in: act_in}
        _, loss = self.session.run([self.train_step, self.loss],
                                   feed_dict=feed_dict)
        return loss

    def eval_loss(self, observations, actions):
        act_in = actions
        if hasattr(self, 'n_actions'):
            acts = np.zeros(shape=(len(actions), self.n_actions))
            for i, act in enumerate(actions):
                acts[i, act] = 1.
            act_in = acts
        feed_dict = {self.obs_input: observations,
                     self.act_in: act_in}
        loss = self.session.run(self.loss, feed_dict=feed_dict)
        loss = loss / np.size(observations, axis = 0)
        return loss

    def save_policy(self, modelpath, itr = -1):
        vs = {}
        for param in self.params:
            vs[param.name] = param
        #vs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
        vs = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
        #print ('saving vars {} to {}'.format(vs, modelpath))
        #saver = tf.train.Saver(vs, max_to_keep = 1)
        if itr >= 0:
            self.saver.save(self.session, modelpath, itr)
        else:
            self.saver.save(self.session, modelpath)

    def load_policy(self, modelpath, latest_checkpoint = True):
        print ('Loading %s' % modelpath)
        #vs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
        vs = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
        #vs = {}
        #for param in self.params:
        #    vs[param.name] = param
        #saver = tf.train.Saver(vs)
        if latest_checkpoint:
            self.saver.restore(self.session, tf.train.latest_checkpoint(modelpath))
        else:
            self.saver.restore(self.session, modelpath)

    def get_dist_vars(self, observation):
        pass

    def entropy(self, observations):
        feed_dict = {self.obs_input: observations}
        return self.session.run(self.avg_entropy, feed_dict=feed_dict)

class GaussianPolicy(NeuralNetworkPolicy):

    def __init__(self, observation_space, action_space, scope=None,
                 learning_rate=1e-04,
                 hidden_sizes=[],
                 train_type='supervised',
                 act_fn=tf.nn.relu,
                 filter_obs=False,
                 seed=None,
                 learn_std=True,
                 entropy_coeff=None,
                 weight_decay=0.0,
                save_dir = None,
                linear_finetune = False):
        self.action_dim = _get_space_size(action_space)
        scope = '' if scope is None else scope
        self.train_type = train_type
        super(GaussianPolicy, self).__init__(observation_space, action_space,
                                             scope=scope,
                                             learning_rate=learning_rate,
                                             hidden_sizes=hidden_sizes,
                                             act_fn=act_fn,
                                             filter_obs=filter_obs,
                                             seed=seed,
                                             entropy_coeff=entropy_coeff,
                                             weight_decay=weight_decay,
                                            linear_finetune = linear_finetune)

        #with self.graph.as_default():

        '''
        self.session = tf.Session()
        obs_dim = _get_space_size(observation_space)
        n_logits = _get_space_size(action_space)
        #self.graph = tf.Graph()
        scope = '' if scope is None else scope
        self.scope = scope
        self.entropy_coeff = entropy_coeff
        self.learning_rate = learning_rate


        #with self.graph.as_default():
        self.params = []
        self.obs_input = tf.placeholder(tf.float32, shape = [None, obs_dim])#, name = 'obs_input')
        #self.obs_input = tflearn.input_data(shape=[None, obs_dim],
        #                                    name='obs_input')
        self.obs_dim = tuple([-1, obs_dim])

        prev = self.obs_input
        init = tf.truncated_normal_initializer(stddev = 1.0)
        with tf.variable_scope(scope):

            #init = tflearn.initializations.truncated_normal(seed=seed
            for idx, size in enumerate(hidden_sizes):
                #prev = tf.layers.dense(prev, size, kernel_initializer = init, name = 'fc%i'%(idx+1), activation = act_fn)
                prev = tf.layers.dense(prev, size, kernel_initializer = init, activation = act_fn)
                self.params.extend([prev])
            #self.logits = tf.squeeze(tf.layers.dense(prev, n_logits, activation = None, kernel_initializer= init, name = 'final'))
            #self.logits = tf.layers.dense(prev, n_logits, activation = None, kernel_initializer= init, name = 'final')

            self.logits = tf.layers.dense(prev, n_logits, activation = None, kernel_initializer= init, name = 'final')
            self.params.extend([self.logits])

            set_trace()
    '''
        with tf.variable_scope(scope):
            self.mean = self.logits

            self.log_std = tf.get_variable(
                'logstd', shape = [1, self.action_dim], initializer=tf.zeros_initializer(), trainable=learn_std)
            self.params.append(self.log_std)
            self.act_in = tf.placeholder(tf.float32, [None, self.action_dim], name = 'Actions')
            #self.act_in = tflearn.input_data(shape=[None, self.action_dim],
            #                                 name='Actions')

            zs = (self.act_in - self.mean) / tf.exp(self.log_std)
            self.log_likelihood = - tf.reduce_sum(self.log_std, axis=-1) - \
                0.5 * tf.reduce_sum(tf.square(zs), axis=-1) - \
                0.5 * self.action_dim * np.log(2 * np.pi)
            self.grad_log_prob = tf.gradients(self.log_likelihood,
                                              self.params)
            ent = tf.log(np.sqrt(2 * np.pi * np.e, dtype=np.float32))
            entropy = self.log_std + ent
            self.avg_entropy = tf.reduce_mean(entropy, axis=-1)
            if self.train_type == 'reinforce':
                self.adv_var =tf.placeholder(tf.float32, [None, 1])
                #self.adv_var = tflearn.input_data(shape=[None, 1])
                self.loss = tf.reduce_sum(tf.multiply(self.adv_var,
                                                      self.log_likelihood))
                optimizer = tf.train.GradientDescentOptimizer(
                    self.learning_rate)
                self.train_step = optimizer.minimize(
                    tf.negative(self.loss))
            elif self.train_type == 'supervised':
                print ('----supervised  training for MLE ------')
                # Loss is sum of neg log likelihoods with optional entropy
                # term. We also compute an avg_loss without the average
                # entropy
                self.loss = tf.reduce_sum(
                    tf.negative(self.log_likelihood))
                # negloglikelihood = tf.negative(self.log_likelihood)
                # self.loss = tf.reduce_sum(negloglikelihood)
                if self.entropy_coeff not in [0.0, None]:
                    print('Coeff %f' % self.entropy_coeff)
                    self.loss -= self.entropy_coeff * tf.reduce_sum(entropy)
                else:
                    print('No Entropy Regularization')
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                #optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
                self.train_step = optimizer.minimize(self.loss)

            self.init_op = tf.global_variables_initializer()

        #self.session = tf.Session(graph=self.graph)
        self.session.run(self.init_op)

        vs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
        #print ('saving vars {} to {}'.format(vs, modelpath))
        self.saver = tf.train.Saver(vs, max_to_keep = 1)

        if save_dir is not None:
            self.psec_pi_dir = save_dir
            print ('saving psec net in {}'.format(self.psec_pi_dir))
            if os.path.isdir(self.psec_pi_dir):
                shutil.rmtree(self.psec_pi_dir)
            os.mkdir(self.psec_pi_dir) 
            clct_ckpt_dir = os.path.join(self.psec_pi_dir, 'pi')
            self.clct_ckpt_file = os.path.join(clct_ckpt_dir, 'pi') 

    def action_prob(self, observation, action):
        pdf = self.pdf(observation, action)
        return pdf

    def action(self, observation, stochastic = True):
        ac, _ = self.get_action(observation, stochastic)
        return ac

    def get_action(self, observation, stochastic=True):
        mean, log_std = self.session.run(
            [self.mean, self.log_std],
            feed_dict={self.obs_input: observation.reshape(self.obs_dim)})
        if not stochastic:
            return mean.flatten(), 1.0
        rnd = np.random.normal(size=self.action_dim)
        # print(rnd[0:4], mean[0:4])
        action = rnd * np.exp(log_std) + mean
        action = action.flatten()
        pdfval = self.pdf(observation, action)
        return action, pdfval

    def pdf(self, observation, action):
        feed_dict = {self.obs_input: observation.reshape(self.obs_dim),
                     self.act_in: action.reshape((1, self.action_dim))}
        log_li = self.session.run(self.log_likelihood, feed_dict=feed_dict)
        prob = np.exp(log_li)
        return prob.flatten()[0]

    def grad_log_policy(self, observation, action, param=0):
        feed_dict = {self.obs_input: observation,
                     self.act_in: action}
        return self.session.run(self.grad_log_pi, feed_dict=feed_dict)

    def reinforce_update(self, observations, actions, advantages):
        feed_dict = {self.obs_input: observations,
                     self.act_in: actions,
                     self.adv_var: advantages.reshape(-1, 1)}
        _, loss = self.session.run([self.train_step, self.loss],
                                   feed_dict=feed_dict)
        return loss

    def get_dist_vars(self, observation):
        mean, log_std = self.session.run(
            [self.mean, self.log_std],
            feed_dict={self.obs_input: observation.reshape(self.obs_dim)})
        return {'mean': mean, 'log_std': log_std}

