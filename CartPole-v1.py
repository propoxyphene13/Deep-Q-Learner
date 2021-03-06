#OpenAI Cartpole


import gym
import tensorflow as tf
import numpy as np        #matrix/math tools


# create class with initialization, experiece, training, and actions
class DeepQLearningAgent(object):
    #self. maintains variables with the object so they dont get reset after method
    #must declare self as it is not implicit in java, also expecting state and action arrguments
    def __init__(self, state_space, action_space):
        #save the arguments passed in to the object as variables that span calls
        self._action_space = action_space
        #figure out the size of state (observation) and action space
        self._dim_state = state_space.shape[0]
        self._dim_action = action_space.n
        #defines the shape of random experiences
        self._batch_size = 400
        #set the q learning gamma
        self._gamma = 0.98
        
        #create previous state/action/reward and initialize variables
        self._prev_state = None
        self._prev_action = None
        self._prev_reward = 0

        #build 4 layer neural network with relu
        #first layer shape depends on the state/observation inputs
        #final layer shape depends on the action space available
        w1 = tf.random_uniform([self._dim_state, 128], -1.0, 1.0)
        w1 = tf.Variable(w1)
        b1 = tf.random_uniform([128], -1.0, 1.0)
        b1 = tf.Variable(b1)

        w2 = tf.random_uniform([128, 128], -1.0, 1.0)
        w2 = tf.Variable(w2)
        b2 = tf.random_uniform([128], -1.0, 1.0)
        b2 = tf.Variable(b2)

        w3 = tf.random_uniform([128, 128], -1.0, 1.0)
        w3 = tf.Variable(w3)
        b3 = tf.random_uniform([128], -1.0, 1.0)
        b3 = tf.Variable(b3)

        w4 = tf.random_uniform([128, self._dim_action], -1.0, 1.0)
        w4 = tf.Variable(w4)
        b4 = tf.random_uniform([self._dim_action], -1.0, 1.0)
        b4 = tf.Variable(b4)
        
        #set up network to find previous values based on action values and masks
        prev_states = tf.placeholder(tf.float32, [None, self._dim_state])
        hidden_1 = tf.nn.relu(tf.matmul(prev_states, w1) + b1)
        hidden_2 = tf.nn.relu(tf.matmul(hidden_1, w2) + b2)
        hidden_3 = tf.nn.relu(tf.matmul(hidden_2, w3) + b3)
        prev_action_values = tf.squeeze(tf.matmul(hidden_3, w4) + b4)
        prev_action_masks = tf.placeholder(tf.float32, [None, self._dim_action])
        prev_values = tf.reduce_sum(tf.mul(prev_action_values, prev_action_masks), reduction_indices=1)
        
        #set up network to determine the value of future actions (Q prime?)
        prev_rewards = tf.placeholder(tf.float32, [None, ])
        next_states = tf.placeholder(tf.float32, [None, self._dim_state])
        hidden_1 = tf.nn.relu(tf.matmul(next_states, w1) + b1)
        hidden_2 = tf.nn.relu(tf.matmul(hidden_1, w2) + b2)
        hidden_3 = tf.nn.relu(tf.matmul(hidden_2, w3) + b3)
        next_action_values = tf.squeeze(tf.matmul(hidden_3, w4) + b4)
        next_values = prev_rewards + self._gamma * tf.reduce_max(next_action_values, reduction_indices=1)
        
        #set up training loss with adam optimizer
        loss = tf.reduce_mean(tf.square(prev_values - next_values))
        training = tf.train.AdamOptimizer(1e-4).minimize(loss)
        
        #assign action and state from 1st net for use in training and determining action to take
        self._tf_action_value_predict = prev_action_values
        self._tf_prev_states = prev_states
        self._tf_prev_action_masks = prev_action_masks
        self._tf_prev_rewards = prev_rewards
        self._tf_next_states = next_states
        self._tf_training = training
        self._tf_loss = loss
        #set up tf session
        self._tf_session = tf.InteractiveSession()
        
        #run tf session & init vars
        self._tf_session.run(tf.initialize_all_variables())
        
        #Double Q setup for epsilon decay and experiences
        self._time = 0
        self._epsilon = 1.0
        self._epsilon_decay_time = 100
        self._epsilon_decay_rate = 0.98
        self._experiences_max = 4000
        self._experiences_num = 0
        self._experiences_prev_states = np.zeros((self._experiences_max, self._dim_state))
        self._experiences_next_states = np.zeros((self._experiences_max, self._dim_state))
        self._experiences_rewards = np.zeros((self._experiences_max))
        self._experiences_actions_mask = np.zeros((self._experiences_max, self._dim_action))
    
    
    #As we're deciding on an action call experiences to save previous results
    def experience(self, prev_state, prev_action, reward, next_state):
        #if we have too much experience we need to pentalize ourself to possibly escape local maxima
        if self._experiences_num >= self._experiences_max:
            #update experiences by dropping some
            a = self._experiences_max * 8 / 10
            b = self._experiences_max - a

            if reward > 0.0:
                idx = np.random.choice(a)
            else:
                idx = np.random.choice(b) + a
        else:
            idx = self._experiences_num

        #increment number of experiences
        self._experiences_num += 1

        self._experiences_prev_states[idx] = np.array(prev_state)
        self._experiences_next_states[idx] = np.array(next_state)
        self._experiences_rewards[idx] = reward
        self._experiences_actions_mask[idx] = np.zeros(self._dim_action)
        self._experiences_actions_mask[idx, prev_action] = 1.0
    
    
    #train the network based on experiences
    def train(self):
        #only train if we have plenty of experiences, return if we dont
        if self._experiences_num < self._experiences_max:
            return
        
        #set ixs as a r random set of experiences in the shape batch size
        ixs = np.random.choice(self._experiences_max, self._batch_size, replace=True)
        
        #create list of action loss and minimized loss
        fatches = [self._tf_loss, self._tf_training]

        #variable feed for placeholders above
        feed = {
            self._tf_prev_states: self._experiences_prev_states[ixs],
            self._tf_prev_action_masks: self._experiences_actions_mask[ixs],
            self._tf_prev_rewards: self._experiences_rewards[ixs],
            self._tf_next_states: self._experiences_next_states[ixs]
        }
        
        #run the tf session and save the loss
        loss, _ = self._tf_session.run(fatches, feed_dict=feed)
    
    
    #take action based on observation rewards (or if we're done)
    def act(self, observation, reward, done):
        #epsilon decay timer
        self._time += 1
        
        #when we reach the decay time interval defined above decay the epsilon
        if self._time % self._epsilon_decay_time == 0:
            self._epsilon *= self._epsilon_decay_rate

        #if we randomly exceed the epsilon, use previous state prediction to determine the max action
        if np.random.rand() > self._epsilon:
            states = np.array([observation])

            action_values = self._tf_action_value_predict.eval(feed_dict={self._tf_prev_states: states})

            action = np.argmax(action_values)
        #otherwise, choose a random action from allowable list (explore)
        else:
            action = self._action_space.sample()

        #If we have a previous state to compare against, create an experience (call experience)
        if self._prev_state is not None:
            self.experience(self._prev_state, self._prev_action, reward, observation)

        #set previous state to observation, previous action to action to action and add reward if we arent done
        self._prev_state = None if done else observation
        self._prev_action = None if done else action
        self._prev_reward = 0 if done else self._prev_reward + reward

        #call training
        self.train()

        return action
#end of class

    
#create envoronment and setup recording, force clears existing training data
env = gym.make('CartPole-v1')
env.monitor.start('/tmp/training_dir', force=True)

#create loop vars, up to 2000 attempts with up to 500 steps/actions in each if we dont fail first
max_episodes = 2000
max_steps = 500
running_reward = []

agent = DeepQLearningAgent(env.observation_space, env.action_space)

#episode attempt loop; xrange slightly faster than range
for episode in xrange(max_episodes):
    #for each episode:
    # observation=initial state as we reset env
    # reward=initialized to 0
    # done=false until the env reports that the episode has ended
    observation = env.reset()
    reward = 0.0
    done = False
    
    #step (observation/action) loop
    for step in xrange(max_steps):
        # penalty for failing: when done flag true and we havent reached the last step in an episode
        # also sets observation array to 0 (zers_like just matches array shape and type)
        if done and step +1 < max_steps:
            reward = -5000.0
            observation = np.zeros_like(observation)
        
        #take action passing observation, reward and done vars
        action = agent.act(observation, reward, done)
        
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
