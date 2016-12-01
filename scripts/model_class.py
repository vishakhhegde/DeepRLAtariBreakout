import cv2
from collections import deque
import gym
import tensorflow as tf
import numpy as np
import sys, os
import random
from utils import *

GAME = 'Breakout-v0'
ACTIONS = 6 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 1000. # timesteps to observe before training
EXPLORE = 1000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.05 # final value of epsilon
INITIAL_EPSILON = 1.0 # starting value of epsilon
REPLAY_MEMORY = 500000 # number of previous transitions to remember
BATCH = 128 # size of minibatch
LEARNING_RATE = 0.00025
K = 1 # only select an action every Kth frame, repeat prev for others

game_state = gym.make(GAME)

class deepRL_model():
	def createBaseNetwork(self):
		# Define weights of the network
		W_conv1 = weight_variable([8, 8, 4, 32])
		b_conv1 = bias_variable([32])

		W_conv2 = weight_variable([4, 4, 32, 64])
		b_conv2 = bias_variable([64])

		W_conv3 = weight_variable([3, 3, 64, 64])
		b_conv3 = bias_variable([64])

		W_fc1 = weight_variable([576, 64])
		b_fc1 = bias_variable([64])

		W_fc2 = weight_variable([64, ACTIONS])
		b_fc2 = bias_variable([ACTIONS])

		s = tf.placeholder("float", [None, 80, 80, 4])

	    # hidden layers
		h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
		h_pool1 = max_pool_2x2(h_conv1)

		h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
		h_pool2 = max_pool_2x2(h_conv2)

#		h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3, 1) + b_conv3)
		print h_pool2
		h_conv3_flat = tf.reshape(h_pool2, [-1, 576])

		h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

		readout = tf.matmul(h_fc1, W_fc2) + b_fc2

		return s, readout, h_fc1

	def trainNetwork(self,s, readout, h_fc1, sess):
	    # define the cost function
		a = tf.placeholder("float", [None, ACTIONS])
		y = tf.placeholder("float", [None])
		readout_action = tf.reduce_sum(tf.mul(readout, a), reduction_indices = 1)
		cost = tf.reduce_mean(tf.square(y - readout_action))
		train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

	    # store the previous observations in replay memory
		D = deque()

	    # printing
		a_file = open("logs_" + GAME + "/readout.txt", 'w')
		h_file = open("logs_" + GAME + "/hidden.txt", 'w')

	    # get the first state by doing nothing and preprocess the image to 80x80x4
		do_nothing = np.zeros(ACTIONS)
		do_nothing[0] = 1

		action = np.where(do_nothing == 1)[0][0]
		x_t, r_0, terminal, _ = game_state.step(action)
		x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)

		ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
		s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2)

	    # saving and loading networks
		saver = tf.train.Saver()
		sess.run(tf.initialize_all_variables())
		checkpoint = tf.train.get_checkpoint_state("saved_networks")
		if checkpoint and checkpoint.model_checkpoint_path:
		    saver.restore(sess, checkpoint.model_checkpoint_path)
		    print "Successfully loaded:", checkpoint.model_checkpoint_path
		else:
		    print "Could not find old network weights"

		epsilon = INITIAL_EPSILON
		t = 0
		while(1):
		    # choose an action epsilon greedily
#			game_state.render()
			readout_t = readout.eval(feed_dict = {s : [s_t]})[0]
			a_t = np.zeros([ACTIONS])
			action_index = 0
			if random.random() <= epsilon or t <= OBSERVE:
				action_index = random.randrange(ACTIONS)
				a_t[action_index] = 1
			else:
				action_index = np.argmax(readout_t)
				a_t[action_index] = 1

	        # scale down epsilon
			if epsilon > FINAL_EPSILON and t > OBSERVE:
				epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

			for i in range(0, K):
	            # run the selected action and observe next state and reward
				action = np.where(a_t == 1)[0][0]
				x_t1_col, r_t, terminal, _ = game_state.step(action)
				if terminal:
					_ = game_state.reset()
				x_t1 = cv2.cvtColor(cv2.resize(x_t1_col, (80, 80)), cv2.COLOR_BGR2GRAY)
				ret, x_t1 = cv2.threshold(x_t1,1,255,cv2.THRESH_BINARY)
				x_t1 = np.reshape(x_t1, (80, 80, 1))
				s_t1 = np.append(x_t1, s_t[:,:,0:3], axis = 2)

	            # store the transition in D
				D.append((s_t, a_t, r_t, s_t1, terminal))
				if len(D) > REPLAY_MEMORY:
					D.popleft()

	        # only train if done observing
			if t > OBSERVE:
	            # sample a minibatch to train on
				minibatch = random.sample(D, BATCH)

	            # get the batch variables
				s_j_batch = [d[0] for d in minibatch]
				a_batch = [d[1] for d in minibatch]
				r_batch = [d[2] for d in minibatch]
				s_j1_batch = [d[3] for d in minibatch]

				y_batch = []
				readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})
				for i in range(0, len(minibatch)):
	                # if terminal only equals reward
					if minibatch[i][4]:
						y_batch.append(r_batch[i])
					else:
						y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

	            # perform gradient step
				train_step.run(feed_dict = {
					y : y_batch,
					a : a_batch,
					s : s_j_batch})

	        # update the old values
			s_t = s_t1
			t += 1

	        # save progress every 10000 iterations
			if t % 10000 == 0:
				saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step = t)

			# print info
			state = ""
			if t <= OBSERVE:
				state = "observe"
			elif t > OBSERVE and t <= OBSERVE + EXPLORE:
				state = "explore"
			else:
				state = "train"
			if t % 10 == 0:
				print "TIMESTEP", t, "/ STATE", state, "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, "/ Q_MAX %e" % np.max(readout_t)
