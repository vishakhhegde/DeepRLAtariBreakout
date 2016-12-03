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

def trainAgent():
	sess = createSess()
	Model =	deepRL_model()
	s, readout, h_fc1 = Model.createBaseNetwork()
#	s, readout, h_fc1 = createNetwork()
	Model.trainNetwork(s, readout, h_fc1, sess)

def testAgent():
	sess = createSess()
	Model = deepRL_model()
	s, readout, h_fc1 = Model.createBaseNetwork()
	Model.testNetwork(s, readout, h_fc1, sess)

def main(runas):
	if runas == 'train':
		trainAgent()
	elif runas == 'test':
		testAgent()
	else:
		print 'Invalid argument'
		return

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--runas', type=str, help='Run train or test')
	args = parser.parse_args()
	runas = args.runas
	main(runas)
