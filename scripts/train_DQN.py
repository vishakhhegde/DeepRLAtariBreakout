import cv2
from collections import deque
import gym
import tensorflow as tf
import numpy as np
import sys, os
import random
from model_class import deepRL_model
from utils import *
import argparse

def createSess():
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
	sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
	return sess

def trainAgent(SAVED_NETWORKS_PATH):
	sess = createSess()
	Model =	deepRL_model(SAVED_NETWORKS_PATH)
	s, readout, h_fc1 = Model.createBaseNetwork()
	Model.trainNetwork(s, readout, h_fc1, sess)

def testAgent(SAVED_NETWORKS_PATH):
	sess = createSess()
	Model = deepRL_model(SAVED_NETWORKS_PATH)
	s, readout, h_fc1 = Model.createBaseNetwork()
	Model.testNetwork(s, readout, h_fc1, sess)

def main(runas, SAVED_NETWORKS_PATH):
	if runas == 'train':
		trainAgent(SAVED_NETWORKS_PATH)
	elif runas == 'test':
		testAgent(SAVED_NETWORKS_PATH)
	else:
		print 'Invalid argument'
		return

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--runas', type=str, help='Run train or test')
	parser.add_argument('--SAVED_NETWORKS_PATH', type=str, help='Path to the saved networks for a given set of training parameters')
	args = parser.parse_args()

	main(args.runas, args.SAVED_NETWORKS_PATH)
