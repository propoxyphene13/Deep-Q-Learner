#OpenAI Cartpole

import gym
import tensorflow as tf
import numpy as np		#matrix/math tools


***** create class with initialization, experiece, training, and actions *****

#create envoronment and setup recording, force clears existing training data
env = gym.make('CartPole-v1')
env.monitor.start('/tmp/training_dir', force=True)

#create loop vars, up to 2000 attempts with up to 500 steps/actions in each
max_episodes = 2000
max_steps = 500
running_reward = []

agent = *****FILLIN*****

#episode attempt loop; xrange slightly faster than range
for episode in xrange(max_episodes):
    #for each episode:
	# observation=initial state as we reset env
	# reward=initialized to 0
	# done=false until the env reports that the episode has ended
	observation = env.reset()
	reward = 0.0
	done = false
	
	#step (observation/action) loop
	for step in xrange(max_steps):
		# penalty for failing: when done flag true and we havent reached the last step in an episode
		# also sets observation array to 0 (zers_like just matches array shape and type)
		if done and step +1 < max_steps:
		    reward = -5000.0
			observation = np.zeros_like(observation)
	    
		#take action passing observation, reward and done vars
		action = *****FILLIN*****
		
		# if we reach the end of the episode, rewards
		if done or step + 1 == max_steps:
			#add the step to the reward array
			running_reward.append(step)
			
			#if we have added over 100 steps to reward array limit array to just the most recent 100
			if len(running_reward) > 100:
			    running_reward = running_reward[-100:]
		
		    # calculate the average steps reached from our list of up to 100 rewards
		    avg_reward = sum(running_reward) / float(len(running_reward))
	
		    #display current episode, step and average reward
			print "{} - {} - {}".format(episode, step, avg_reward)
			
			#not sure what this break is for
			break
		
		# run one timestep of environment with the action passed in and observation/reward/done/info returned
		observation, reward, done, _ = env.step(action)
		
	# end of step loop
	
#end of episode loop
#close environment
env.monitor.close()
