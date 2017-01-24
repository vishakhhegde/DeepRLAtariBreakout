# Deep Reinforcement Learning (DQN) for Atari Breakout
In this project, we re - implement DeepMind's Deep Q Network (DQN), with some additional bells and whistles. This is our course project for Stanford's CS221: Artificial Intelligence. We use the Atari environment provided by OpenAI in the OpenAI gym. 

When we reviewed some code bases to train a network on the OpenAI environment, we found that most of them are not very readable and involves a lot of abstractions. While abstractions are useful to add additional bells and whistles to the DQN algorithm, it does not help much in understanding the underlying concepts of DQN. We hope that this codebase is more readable than many others.

## OpenAI Atari environment
The Atari environment we used can be accessed at https://gym.openai.com/envs#atari. We specifically use Breakout-v0 in our programs. Make sure you install the openAI gym python library (at https://gym.openai.com/docs) before continuing.

Basically, the environment takes in an action (a number between 0 - 5) and outputs the following:
1. Next frame of the game
2. Reward for taking the action
3. A boolean indicating end of the game (after while, you will need to reset the environment to play the game again)
4. 'Info': Not useful for our purposes. Can be useful for debugging.

## Pipeline
All we know at the beginning is that we have an environment where we can take certain actions. The actions result in a new game state and gives us a reward for the action taken. So, the only feedback we have is the reward. The goal is to take the best possible action for any given game screen. i.e., get the most optimal policy. Q-learning is a way to learn the optimal policy. We use Q - learning with function approximation (where our function is the deep network). The function (which is smooth and differentiable)hopefully will help us take good actions even for unseen states.

