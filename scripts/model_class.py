import cv2
from collections import deque
import gym
import tensorflow as tf
import numpy as np
import sys, os
import random
from utils import *
import copy
from training_parameters import *
from shutil import copyfile

game_state = gym.make(GAME)
NUM_TEST_GAMES = 100
K = 1 # only select an action every Kth frame, repeat prev for others
TEST_EPSILON = 0.05

class deepRL_model():
	def __init__(self, SAVED_NETWORKS_PATH, networkName):		
		self.SAVED_NETWORKS_PATH = SAVED_NETWORKS_PATH
		self.networkName = networkName

	def createBaseNetwork(self):
		if self.networkName == '3LayerConv':
		# Define weights of the network
			W_conv1 = weight_variable([8, 8, 4, 32])
			b_conv1 = bias_variable([32])

			W_conv2 = weight_variable([4, 4, 32, 64])
			b_conv2 = bias_variable([64])

			W_conv3 = weight_variable([3, 3, 64, 64])
			b_conv3 = bias_variable([64])

			W_fc1 = weight_variable([6400, 512])
			b_fc1 = bias_variable([512])

			W_fc2 = weight_variable([512, NUM_ACTIONS])
			b_fc2 = bias_variable([NUM_ACTIONS])

			s = tf.placeholder("float", [None, 80, 80, 4])

		    # hidden layers
			conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)

			conv2 = tf.nn.relu(conv2d(conv1, W_conv2, 2) + b_conv2)

			conv3 = tf.nn.relu(conv2d(conv2, W_conv3, 1) + b_conv3)

			conv3_flat = tf.reshape(conv3, [-1, 6400])

			h_fc1 = tf.nn.relu(tf.matmul(conv3_flat, W_fc1) + b_fc1)

			Qvalues = tf.matmul(h_fc1, W_fc2) + b_fc2

			return s, Qvalues, h_fc1

		elif self.networkName == '1LayerFC':
			W_fc1 = weight_variable([6400, 512])
			b_fc1 = bias_variable([512])

			W_fc2 = weight_variable([512, NUM_ACTIONS])
			b_fc2 = bias_variable([NUM_ACTIONS])

			s = tf.placeholder("float", [None, 80, 80, 4])

			# The actual network
			pool1 = max_pool_2x2(s)
			flat = tf.reshape(pool1, [-1, 6400])
			h_fc1 = tf.nn.relu(tf.matmul(flat, W_fc1) + b_fc1)
			Qvalues = tf.matmul(h_fc1, W_fc2) + b_fc2
			return s, Qvalues, h_fc1


		else:
			print 'Not a valid network name'
			return


	def trainNetwork(self,s, Qvalues, h_fc1, sess):
	    # define the cost function
		ensure_dir_exists(self.SAVED_NETWORKS_PATH)
		dest_path = os.path.join(self.SAVED_NETWORKS_PATH,'training_parameters.py')
		copyfile('training_parameters.py', dest_path)

		f = open(dest_path, 'a')
		f.write('networkName = {}'.format(self.networkName))
		f.close()

		a = tf.placeholder("float", [None, NUM_ACTIONS])
		y = tf.placeholder("float", [None])
		readout_action = tf.reduce_sum(tf.mul(Qvalues, a), reduction_indices = 1)
		cost = tf.reduce_mean(tf.square(y - readout_action))
		train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

	    # store the previous observations in replay memory
		D = deque()

	    # get the first state by doing nothing and preprocess the image to 80x80x4
		do_nothing = np.zeros(NUM_ACTIONS)
		do_nothing[0] = 1

		action = np.where(do_nothing == 1)[0][0]
		x_t, r_0, terminal, _ = game_state.step(action)
		x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)

		ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
		s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2)

	    # saving and loading networks
		saver = tf.train.Saver()
		sess.run(tf.initialize_all_variables())
		checkpoint = tf.train.get_checkpoint_state(self.SAVED_NETWORKS_PATH)
		checkpoint_IterNum = 0
		if checkpoint and checkpoint.model_checkpoint_path:
			checkpoint_IterNum = int(checkpoint.model_checkpoint_path.split('-')[-1])
			print checkpoint_IterNum
			saver.restore(sess, checkpoint.model_checkpoint_path)
			print "Successfully loaded:", checkpoint.model_checkpoint_path
		else:
		    print "Could not find old network weights"

		epsilon = INITIAL_EPSILON
		t = 0
		while(1):
		    # choose an action epsilon greedily
#			game_state.render()
			readout_t = Qvalues.eval(feed_dict = {s : [s_t]})[0]
			a_t = np.zeros([NUM_ACTIONS])
			action_index = 0

			if random.random() <= epsilon or t <= OBSERVE:
				action_index = random.randrange(NUM_ACTIONS)
				isRandom = 1
				a_t[action_index] = 1
			else:
				action_index = np.argmax(readout_t)
				isRandom = 0
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
	            # If the action was random, the reward is r_t
	            # If the action was not random, the reward is scaled by 5 to give it a higher weight
				if not isRandom:
					D.append((s_t, a_t, 5*r_t, s_t1, terminal))
				else:
					D.append((s_t, a_t, r_t, s_t1, terminal))

				if len(D) > REPLAY_MEMORY:
					D.popleft()

			if t > OBSERVE:
	            # minibatch for training
				minibatch = random.sample(D, BATCH)

	            # get the batch variables
				s_j_batch = [d[0] for d in minibatch]
				a_batch = [d[1] for d in minibatch]
				r_batch = [d[2] for d in minibatch]
				s_j1_batch = [d[3] for d in minibatch]

				y_batch = []
				readout_j1_batch = Qvalues.eval(feed_dict = {s : s_j1_batch})
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

	        # save progress every 15000 iterations
			if t % 15000 == 0:
				saver.save(sess, self.SAVED_NETWORKS_PATH + '/' + GAME + '-dqn', global_step = t + checkpoint_IterNum)

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

	def testNetwork(self, s, Qvalues, h_fc1, sess):
		# Initialization of the state
		do_nothing = np.zeros(NUM_ACTIONS)
		do_nothing[0] = 1

		action = np.where(do_nothing == 1)[0][0]
		x_t, r_0, terminal, _ = game_state.step(action)
		x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)

		ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
		s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2)
		####################################

		# Load the meta file and the checkpoint #
		saver = tf.train.Saver()

		checkpoint = tf.train.get_checkpoint_state(self.SAVED_NETWORKS_PATH)
		if checkpoint and checkpoint.model_checkpoint_path:
			checkpoint_IterNum = checkpoint.model_checkpoint_path.split('-')[-1]
			saver.restore(sess, checkpoint.model_checkpoint_path)
			print "Successfully loaded:", checkpoint.model_checkpoint_path
		else:
			print "Could not find old network weights"
			return
		###############################################

		# Play the game a fixed number of steps
		i = 0
		game_score = 0
		allGameScores = []
		f = open(os.path.join(self.SAVED_NETWORKS_PATH, 'test_results_' + checkpoint_IterNum + '.txt'), 'w')
		f.write('GAME_SCORES\n')
		random.seed(1)
		while i < NUM_TEST_GAMES:
#			game_state.render()
			if not TEST_EPSILON == 1.0:
				readout_t = Qvalues.eval(feed_dict = {s : [s_t]})[0]
				if random.random() <= TEST_EPSILON:
					action = random.randrange(NUM_ACTIONS)
				else:
					action = np.argmax(readout_t)
			else:
				action = random.randrange(NUM_ACTIONS)

			for j in range(0, K):
	            # run the selected action and observe next state and reward
				x_t1_col, r_t, terminal, _ = game_state.step(action)
				game_score += r_t
				if terminal:
					_ = game_state.reset()
					print 'game_score = ', game_score
					allGameScores.append(game_score)
					f.write(str(game_score) + '\n')
					game_score = 0
					i += 1
				x_t1 = cv2.cvtColor(cv2.resize(x_t1_col, (80, 80)), cv2.COLOR_BGR2GRAY)
				ret, x_t1 = cv2.threshold(x_t1,1,255,cv2.THRESH_BINARY)
				x_t1 = np.reshape(x_t1, (80, 80, 1))
				s_t1 = np.append(x_t1, s_t[:,:,0:3], axis = 2)

			s_t = copy.deepcopy(s_t1)
		avg_gameScore = sum(allGameScores)/NUM_TEST_GAMES
		max_gameScore = max(allGameScores)
		f.write('Average GameScore = {} \n'.format(avg_gameScore))
		f.write('Max GameScore = {} \n'.format(max_gameScore))
		f.close()

