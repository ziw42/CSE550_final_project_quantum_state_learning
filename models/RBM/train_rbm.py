########## Multinomial RBM class  ##########
### Author: Juan Carrasquilla
###########################################

from __future__ import print_function
import argparse
import tensorflow as tf
import itertools as it
from rbm import RBM
import numpy as np
import math
import os
import sys 
# CLI arguments
parser = argparse.ArgumentParser(description='Train multinomial RBM')
parser.add_argument('--p', type=float, required=True, help='Noise value p')
parser.add_argument('--L', type=int, required=True, help='Number of qubits (visible sites)')
parser.add_argument('--num_state_vis', type=int, required=True, help='Local outcomes per site (K)')
parser.add_argument('--num_hidden', type=int, default=8, help='Number of hidden units')
parser.add_argument('--nsteps', type=int, default=360000, help='Training steps')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
parser.add_argument('--num_gibbs', type=int, default=2, help='CD/PCD Gibbs steps')
parser.add_argument('--num_samples', type=int, default=100, help='Chains in PCD/CD when CD=True')
parser.add_argument('--cd', action='store_true', help='Use CD_k (default True); omit to use PCD_k')
args = parser.parse_args()

p = args.p
L                   = args.L
num_state_vis       = args.num_state_vis
num_state_hid       = 2
num_visible         = L
num_hidden          = args.num_hidden
nsteps              = args.nsteps
learning_rate_start = 3e-5
bsize               = args.batch_size
num_gibbs           = args.num_gibbs
num_samples         = args.num_samples
CD                  = True if args.cd or True else False



### Function to save weights and biases to a parameter file ###
def save_parameters(sess, rbm,epoch):
    weights, visible_bias, hidden_bias = sess.run([rbm.weights, rbm.visible_bias, rbm.hidden_bias])
    
    # Save into a directory tagged by hyperparameters
    parameter_dir = 'RBM_parameters' + '_p' + str(p) + '_L' + str(L) + '_K' + str(num_state_vis)
    if not(os.path.isdir(parameter_dir)):
      os.mkdir(parameter_dir)
    parameter_file_path =  '%s/parameters_nH%d_L%d' %(parameter_dir,num_hidden,L)
    parameter_file_path += '_p' + str(p)
    parameter_file_path += '_K' + str(num_state_vis)
    parameter_file_path += '_epoch' + str(epoch)
    np.savez_compressed(parameter_file_path, weights=weights, visible_bias=visible_bias, hidden_bias=hidden_bias)

class Placeholders(object):
    pass

class Ops(object):
    pass

weights      = None  #weights None gets them initialized randomly
visible_bias = None  #visible bias None gets them initialized at zero
hidden_bias  = None  #hidden bias None gets them initialized at zero

# Load the MC configuration training data:
trainFileName = 'data/train.txt'
xtrain        = np.loadtxt(trainFileName)
ept           = np.random.permutation(xtrain) # random permutation of training data
iterations_per_epoch = xtrain.shape[0] // bsize  

# Initialize the RBM 
rbm = RBM(num_hidden=num_hidden, num_visible=num_visible,num_state_vis=num_state_vis, num_state_hid=num_state_hid, weights=weights, visible_bias=visible_bias,hidden_bias=hidden_bias, num_samples=num_samples) 

# Initialize operations and placeholders classes
ops          = Ops()
placeholders = Placeholders()
placeholders.visible_samples = tf.placeholder(tf.float32, shape=(None, num_visible*num_state_vis), name='v') # placeholder for training data

total_iterations = 0 # starts at zero 
ops.global_step  = tf.Variable(total_iterations, name='global_step_count', trainable=False)
learning_rate    = tf.train.exponential_decay(
    learning_rate_start,
    ops.global_step,
    100 * xtrain.shape[0]/bsize,
    1.0 # decay rate = 1 means no decay
)
 
if CD==True:
    cost  = rbm.neg_log_likelihood_grad(placeholders.visible_samples,placeholders.visible_samples, num_gibbs=num_gibbs)
    bsize = rbm.num_samples
else: # performs PCD 
    cost  = rbm.neg_log_likelihood_grad(placeholders.visible_samples, num_gibbs=num_gibbs)

   
optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-2)
ops.lr    = learning_rate
ops.train = optimizer.minimize(cost, global_step=ops.global_step)
ops.init  = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())



with tf.Session() as sess:
  sess.run(ops.init)
  # Save initial (untrained) parameters as epoch 0 for downstream analysis
  save_parameters(sess, rbm, 0)
  
  bcount      = 0  #counter
  epochs_done = 1  #epochs counter
  #sys.exit()
  for ii in range(nsteps):
    if bcount*bsize+ bsize>=xtrain.shape[0]:
      bcount = 0
      ept    = np.random.permutation(xtrain)

    batch     =  ept[ bcount*bsize: bcount*bsize+ bsize,:]
    bcount    += 1
    feed_dict =  {placeholders.visible_samples: batch}
    
    _, num_steps = sess.run([ops.train, ops.global_step], feed_dict=feed_dict)

    if num_steps % iterations_per_epoch == 0:
      print ('Epoch = %d' % epochs_done)
      save_parameters(sess, rbm,epochs_done)
      epochs_done += 1