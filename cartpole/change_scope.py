import numpy as np
import tensorflow as tf
from random import shuffle
from pdb import set_trace
from numpy import linalg as LA
from policy import NNPolicy

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--pi_weightfile', default = str, help='weight file for behavior policy')
parser.add_argument('--outdir', default = str, help='weight file for behavior policy')

FLAGS = parser.parse_args()

#ckpt_split = FLAGS.pi_weightfile.split('/')
#ckpt = ('/'.join(ckpt_split + [ckpt_split[1]])) + '-0'
ckpt = FLAGS.pi_weightfile

vars = tf.contrib.framework.list_variables(ckpt)
with tf.Graph().as_default(), tf.Session().as_default() as sess:

  new_vars = []
  for name, shape in vars:
    v = tf.contrib.framework.load_variable(ckpt, name)
    if 'oldpi' in name or 'vf' in name:
        continue
    print (name)
    print (tf.Variable(v))
    #new_vars.append(tf.Variable(v, name=name.replace('pi/pol', 'pgpol')))
    new_vars.append(tf.Variable(v, name=name.replace('pgpol', 'finetunepgpol')))
    #new_vars.append(tf.Variable(v, name=name.replace('finetunepgpol','finetunepgpol2' )))

  print (new_vars)
  if len(new_vars) > 0:
    saver = tf.train.Saver(new_vars)
    sess.run(tf.global_variables_initializer())
    #set_trace()
    #saver.save(sess, 'policies_acro/ppo_acro_2_dup/ppo_acro_2_dup', global_step = 0)

