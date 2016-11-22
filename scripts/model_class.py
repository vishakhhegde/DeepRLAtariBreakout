import tensorflow as tf
import numpy as np
from utils import *

GAME = 'Breakout-v0'
ACTIONS = 6 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 500. # timesteps to observe before training
EXPLORE = 500. # frames over which to anneal epsilon
FINAL_EPSILON = 0.05 # final value of epsilon
INITIAL_EPSILON = 1.0 # starting value of epsilon
REPLAY_MEMORY = 590000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
K = 1 # only select an action every Kth frame, repeat prev for others

class deepRL_model():
	def createBaseNetwork(self):
		# Define weights of the network
		W_conv1 = weight_variable([8, 8, 4, 32])
		b_conv1 = bias_variable([32])

		W_conv2 = weight_variable([4, 4, 32, 64])
		b_conv2 = bias_variable([64])

		W_conv3 = weight_variable([3, 3, 64, 64])
		b_conv3 = bias_variable([64])

		W_fc1 = weight_variable([1600, 512])
		b_fc1 = bias_variable([512])

		W_fc2 = weight_variable([512, ACTIONS])
		b_fc2 = bias_variable([ACTIONS])

		s = tf.placeholder("float", [None, 80, 80, 4])

	    # hidden layers
		h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
		h_pool1 = max_pool_2x2(h_conv1)

		h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)

		h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)

		h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

		h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

		readout = tf.matmul(h_fc1, W_fc2) + b_fc2

		return s, readout, h_fc1