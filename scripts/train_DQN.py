import cv2
from collections import deque
import gym
import tensorflow as tf
import numpy as np
import sys, os
import random
from model_class import deepRL_model
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
K = 1 # only select an action every Kth frame, repeat prev for others

game_state = gym.make(GAME)

def playGame():
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.46)
	sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
	Model =	deepRL_model()
	s, readout, h_fc1 = Model.createBaseNetwork()
#	s, readout, h_fc1 = createNetwork()
	Model.trainNetwork(s, readout, h_fc1, sess)

def main():
	playGame()

if __name__ == "__main__":
	main()
