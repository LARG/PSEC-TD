import numpy as np
import tensorflow as tf
from random import shuffle
from pdb import set_trace
from numpy import linalg as LA

class NN(object):
    def __init__(self,
                 state_dims,
                 num_actions,
                 num_hidden_layers,
                 num_neurons,
                 act_fn,
                 alpha,
                 ckpt = False,
                linear_finetune = False,
                nn_finetune = False,
                scope_name = 'pol',
                opt = 'gd',
                reg_param = 0):#2e-2):

        self.sess = tf.Session()
        self.state_dims = state_dims
        self.num_actions = num_actions
        self.global_update_counter = 1
        self.reg_param = reg_param

        train_scratch = (not linear_finetune) and (not nn_finetune)

        print ('state dims {}'.format(state_dims))
        print ('action dims {}'.format(num_actions))
        print ('lr {}'.format(alpha))
        print ('num hidden {}'.format(num_hidden_layers))
        print ('num neurons {}'.format(num_neurons))
        print ('act fn {}'.format(act_fn))
        print ('reg param  {}'.format(reg_param))

        num_units = num_neurons
        if act_fn == 0:
            activ = tf.nn.tanh
        elif act_fn == 1:
            activ = tf.nn.relu
        elif act_fn == 2:
            activ = tf.nn.sigmoid
        elif act_fn == 3:
            def lrelu(x):
                return tf.nn.leaky_relu(x, alpha = 0.01)
            activ = lrelu
        init = tf.contrib.layers.xavier_initializer()
        #init = tf.truncated_normal_initializer(stddev = 0.5)
        
        self.x = tf.placeholder(tf.float32, [None, state_dims])
        self.out = self.x
        with tf.variable_scope(scope_name):
            hidden_training = train_scratch or nn_finetune
            print ('training hidden layers of PSEC {}'.format(hidden_training))
            for _ in range(num_hidden_layers):
                self.out = tf.layers.dense(self.out, num_units, activation = activ, kernel_initializer = init, trainable = hidden_training)
            last_training = train_scratch or nn_finetune or linear_finetune
            print ('training last linear layer of PSEC {}'.format(last_training))
            self.out = tf.squeeze(tf.layers.dense(self.out, num_actions, activation = None, kernel_initializer = init, trainable = last_training))
            self.scope = tf.get_variable_scope().name

        self.y = tf.placeholder(tf.float32, [None, num_actions])

        self.cross = tf.nn.softmax_cross_entropy_with_logits(logits = self.out, labels = self.y)
        self.loss = tf.reduce_mean(self.cross)

        self.lr_decay_freq = 100
        self.lr_ph = tf.placeholder(tf.float32, shape = [])
        self.lr = alpha

        if opt == 'gd':
            print ('gd opt')
            self.lr_decay_rate = 0.9
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.lr_ph)
        elif opt == 'adam':
            print ('adam opt')
            self.lr_decay_rate = 1.
            self.optimizer = tf.train.AdamOptimizer(learning_rate = self.lr_ph, beta1 = 0.9, beta2 = 0.999)
        
        #tvs = tf.trainable_variables()
        tvs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
        print (tvs)

        self.reg_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in tvs])
        self.total_loss = self.reg_param * self.reg_loss + self.loss

        self.gradients = self.optimizer.compute_gradients(self.total_loss, tvs)
        #self.gvals = [g[0] for g in self.gradients]
        #self.gradients = [(g, v[1]) for g, v in zip(self.gvals, self.gradients)]
        self.gradients = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in self.gradients]
        #self.gradients = [(tf.clip_by_norm(grad, 1.), var) for grad, var in self.gradients]
        self.train_op = self.optimizer.apply_gradients(self.gradients)
        #self.train_op = self.optimizer.minimize(self.total_loss)

        #self.saver = tf.train.Saver(var_list=tvs, max_to_keep = 1)
        self.saver = tf.train.Saver(var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope), max_to_keep = 1)

        if ckpt:
            self.saver.restore(self.sess, ckpt)
            if nn_finetune or linear_finetune:
                self.sess.run(tf.variables_initializer(self.optimizer.variables()))
        else:
            self.sess.run(tf.global_variables_initializer())
            #self.sess.run(tf.initialize_variables(tvs))

        print ('================ POLICY STRUCT ckpt {} and tvs {}, linear fine tune {} and nn finetune {}'.format(ckpt, tvs, linear_finetune, nn_finetune))
        self.sess.run(tf.assert_variables_initialized(var_list = tvs))

    def save(self, dir_path, num_iter):
        self.saver.save(self.sess, dir_path, global_step=num_iter)
    
    def load(self, dir_path):
        self.saver.restore(self.sess, dir_path)

    def train(self, s, y):

        '''
        grads, self.tr_loss = self.sess.run([self.gvals, self.loss], {self.x: s, self.y: y})
        grads = [v / LA.norm(v) if LA.norm(v) != 0 else v for v in grads]
        fdict = {g: v for g, v in zip(self.gvals, grads)}
        fdict[self.lr_ph] = self.lr
        self.sess.run([self.train_op], fdict)
        '''
        grads, _, self.tr_loss = self.sess.run([self.gradients, self.train_op, self.loss],
                                    {self.x: s, self.y: y, self.lr_ph: self.lr})

        if (self.global_update_counter % self.lr_decay_freq == 0):
            self.lr = self.lr * self.lr_decay_rate
            print ('=============== itr {} updated PSEC lr to {} ============== '.format(self.global_update_counter, self.lr))
        self.global_update_counter += 1

    def validate(self, s, y):
        self.val_loss = self.sess.run(self.loss, {self.x: s, self.y: y})

    def action(self, s):
        out = self.sess.run(self.out, {self.x: s})
        out = np.exp(out) / np.sum(np.exp(out), keepdims = True)
        return np.random.choice(out.shape[0], size = 1, p = out, replace = False)[0]

    def action_prob(self, s, act):
        out = self.sess.run(self.out, {self.x: s})
        # returns [None, 1], in our case, just evaluating one s at a time
        out = (np.exp(out) / np.sum(np.exp(out), keepdims = True))
        return out[act]

class LinearPolicy(object):
    def __init__(self,
                 state_dims,
                 num_actions,
                 alpha):
        self.sess = tf.Session()
        self.state_dims = state_dims
        self.num_actions = num_actions

        self.x = tf.placeholder(tf.float32, [None, state_dims])
        weight_initializer = tf.contrib.layers.xavier_initializer()
        w = tf.get_variable('piweights', shape = [state_dims, num_actions], dtype = 'float', initializer = weight_initializer)
        b = tf.get_variable('pibias', shape = [num_actions], dtype = 'float', initializer = weight_initializer)
        out = tf.matmul(self.x, w) + b
        self.out = out
        #self.out = tf.nn.softmax(out)
        self.y = tf.placeholder(tf.float32, [None, num_actions])

        self.cross = tf.nn.softmax_cross_entropy_with_logits(logits = self.out, labels = self.y)
        self.loss = tf.reduce_mean(self.cross)
        self.optimizer = tf.train.AdamOptimizer(learning_rate = alpha, beta1 = 0.9, beta2 = 0.999)
        self.train_op = self.optimizer.minimize(self.loss)
        
        var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.sess.run(tf.initialize_variables(var_lists))

    def action(self, s):
        out = self.sess.run(self.out, {self.x: s})
        out = np.exp(out) / np.sum(np.exp(out), keepdims = True)[0]
        return np.random.choice(out.shape[0], size = 1, p = out, replace = False)

    def action_prob(self, s, act):
        out = self.sess.run(self.out, {self.x: s})
        # returns [None, 1], in our case, just evaluating one s at a time
        out = (np.exp(out) / np.sum(np.exp(out), keepdims = True))[0]
        return out[act]

    def train(self, s, y):
        _, self.tr_loss = self.sess.run([self.train_op, self.loss], {self.x: s, self.y: y})

    def validate(self, s, y):
        self.val_loss = self.sess.run(self.loss, {self.x: s, self.y: y})

    def get_weights(self, outfile = None):
        var = [v for v in tf.trainable_variables()]
        w = var[0]
        b = var[1]
        W = np.array(self.sess.run(w))
        b = np.array(self.sess.run(b))
        if outfile is not None:
            with open(outfile, 'w') as w:
                w.write('W\n') 
                for i in range(len(W)):
                    for j in range(len(W[i])):
                        if j > 0:
                            w.write(',')
                        w.write(str(W[i][j]))
                    w.write('\n')
                w.write('B\n')
                for i in range(len(b)):
                    if i > 0:
                        w.write(',')
                    w.write(str(b[i]))
                w.write('\n\n')                 

