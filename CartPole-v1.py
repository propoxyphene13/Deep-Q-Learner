#OpenAI Cartpole

import gym
import tensorflow as tf
import numpy as np		#matrix/math tools


env = gym.make('CartPole-v1')
env.monitor.start('training_dir', force=True)

max_episodes = 2000
max_steps = 500
running_reward = []
