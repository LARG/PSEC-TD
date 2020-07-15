import numpy as np
from algo import ValueFunctionWithApproximation
from numpy import linalg as LA

from pdb import set_trace
import tensorflow as tf

class ValueFunctionWithNN(ValueFunctionWithApproximation):
    def __init__(self,
                 state_dims,
                alpha,
                num_hidden_layers,
                num_neurons,
                act_fn):
        """
        state_dims: the number of dimensions of state space
        """
        self.sess = tf.Session()

        self.state_dims = state_dims
        self.global_update_counter = 1

        print ('lr {}'.format(alpha))
        print ('num hidden {}'.format(num_hidden_layers))
        print ('num neurons {}'.format(num_neurons))
        print ('act fn {}'.format(act_fn))
        #self.x = tf.placeholder(tf.float32, [1, state_dims])
        self.x = tf.placeholder(tf.float32, [None, state_dims])

        num_units = num_neurons
        if act_fn == 0:
            activ = tf.nn.tanh
        elif act_fn == 1:
            activ = tf.nn.relu
        elif act_fn == 2:
            activ = tf.nn.sigmoid     
    
        init = tf.contrib.layers.xavier_initializer()
        #init = tf.truncated_normal_initializer(stddev = 0.1)
        #init = tf.contrib.layers.variance_scaling_initializer()
        self.out = self.x
        with tf.variable_scope('vf'):
            for l_num in range(num_hidden_layers):
                self.out = tf.layers.dense(self.out, num_units, activation = activ, kernel_initializer = init)
            self.out = tf.squeeze(tf.layers.dense(self.out, 1, activation = None, kernel_initializer= init))
            self.scope = tf.get_variable_scope().name
        
        #self.y = tf.placeholder(tf.float32, [1, 1])
        self.y = tf.placeholder(tf.float32, [None])
        #self.rho = tf.placeholder(tf.float32, [1, 1])
        self.rho = tf.placeholder(tf.float32, [None])

        # applying PSEC correction
        self.cost = 0.5 * self.rho * tf.losses.mean_squared_error(labels=self.y, predictions=self.out, reduction = 'none') 

        self.lr_decay_freq = 10#50#100
        self.lr_decay_rate = 0.95#5
        self.lr_ph = tf.placeholder(tf.float32, shape = [])
        self.lr_fixed = alpha
        self.lr = alpha

        # getting variables
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.lr_ph)
        #self.optimizer = tf.train.AdamOptimizer(learning_rate = self.lr_ph, beta1 = 0.9, beta2 = 0.999)
        tvs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
        print (tvs)

        # gradient accumulation
        # maintain accum array
        self.accum_vals = None
        
        # gets gradients
        self.gradients = self.optimizer.compute_gradients(self.cost, tvs)
        self.gvals = [g[0] for g in self.gradients]
        self.accum_grads = [(g, v[1]) for g, v in zip(self.gvals, self.gradients)]

        # clip the gradient values for each variable
        #self.accum_grads = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in self.accum_grads]
        #self.accum_grads = [(tf.divide(grad, tf.norm(grad)), var) for grad, var in self.accum_grads]
                        
        # training
        self.train_op = self.optimizer.apply_gradients(self.accum_grads)
        
        # weights init
        self.sess.run(tf.global_variables_initializer())

    def check_weights(self):
        var = [v for v in tf.trainable_variables(scope = 'vf')]
        w = var[0]
        b = var[1]
        W = np.array(self.sess.run(w))
        b = np.array(self.sess.run(b))
        print ('weights')
        print (W)
        print (b)
    
    def reset(self):
        self.sess.run(tf.initialize_variables(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)))
        #self.sess.run(tf.global_variables_initializer())
        self.lr = self.lr_fixed
        self.accum_vals = None
        self.global_update_counter = 1

    def __call__(self,s):
        s = np.array(s)
        s = np.reshape(s, (-1, self.state_dims))
        out = self.sess.run(self.out, {self.x: s})#[0][0]
        return None, out

    def get_curr_val(self, s):
        s = np.array(s)
        s = np.reshape(s, (-1, self.state_dims))
        out = self.sess.run(self.out, {self.x: s})#[0]#[0]
        return out

    def update(self):

        #print (' new accum approach ')
        # verified that this val does not include the lr
        #print ('before accum')
        #print (self.accum_vals)
        #tn = [LA.norm(v) for v in self.accum_vals]
        #print ('norms')
        #print (tn)
        self.accum_vals = [v / LA.norm(v) if LA.norm(v) != 0 else v for v in self.accum_vals]

        #print ('normed accum vals')
        #print (self.accum_vals)
        fdict = {g: v for g,v in zip(self.gvals, self.accum_vals)}

        fdict[self.lr_ph] = self.lr

        self.sess.run(self.train_op, feed_dict=fdict)

        if (self.global_update_counter % self.lr_decay_freq == 0):
            self.lr = self.lr * self.lr_decay_rate
            print ('============ itr {} updated lr to {} ============ '.format(self.global_update_counter, self.lr))

        self.global_update_counter += 1

        #print ('accum grads')
        #print (self.accum_vals)
        #l = self.sess.run([self.lr_ph], feed_dict = fdict)
        #print (l)

    #def update(self, alpha, G, s_tau):
    def accum_batch(self, alpha, G, s_tau, a_tau, rho_tau):
        s_tau = np.stack(s_tau)
        cost, vg, out = self.sess.run([self.cost, self.gvals, self.out], {self.x: s_tau, self.y: G, self.rho: rho_tau, self.lr_ph: self.lr})
        self.accum_vals = vg
    
    def flush_accum(self):
        self.accum_vals = None
        # self.sess.run(self.zero_ops)


    

