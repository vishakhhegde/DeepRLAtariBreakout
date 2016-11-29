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
OBSERVE = 500. # timesteps to observe before training
EXPLORE = 500. # frames over which to anneal epsilon
FINAL_EPSILON = 0.05 # final value of epsilon
INITIAL_EPSILON = 1.0 # starting value of epsilon
REPLAY_MEMORY = 200000 # number of previous transitions to remember
BATCH = 50 # size of minibatch
K = 1 # only select an action every Kth frame, repeat prev for others

game_state = gym.make(GAME)

def playGame():
	sess = tf.InteractiveSession()
	Model =	deepRL_model()
	s, readout, h_fc1 = Model.createBaseNetwork()
#	s, readout, h_fc1 = createNetwork()
	Model.trainNetwork(s, readout, h_fc1, sess)

def main():
	playGame()

if __name__ == "__main__":
	main()
