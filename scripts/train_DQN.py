import cv2
from collections import deque
import gym
import tensorflow as tf
import numpy as np
import sys, os
import random
from model_class import deepRL_model
from utils import *

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
