import tensorflow as tf
import numpy as np
import sys, os
import gym
#from model_class import *
import pickle

env = gym.make("Breakout-v0")
observation = env.reset()
game_mode = 'rgb_array'
total_reward = 0
base_folder = '/Users/vishakhhegde/DeepRLAtariBreakout/pickle_states'
BATCH_SIZE = 32
NUM_EPISODES = 100

# Collect S, A, R, S'
S = []
for episode in range(NUM_EPISODES):
	_ = env.reset()
	for stepNum in range(BATCH_SIZE):
	  X = env.render(mode = game_mode)
	  A = env.action_space.sample() # your agent here (this takes random actions)
	  observation, R, done, info = env.step(A)
	  if done:
	  	break
	  print done, R
#	  print observation.shape
	  X_prime = env.render(mode = game_mode)
#	  S.append({'X': X, 'A': A, 'R': R, 'X_prime': X_prime})
	
#	S_filename = 'episode' + str(episode) + '.pkl'
#	S_filePath = os.path.join(base_folder, S_filename)
#	pickle.dump(S, open(S_filePath, 'w'))