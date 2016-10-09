#OpenAI Cartpole


import gym
import tensorflow as tf
import numpy as np        #matrix/math tools
#import nplot              #for Q plotting
import random

    

#Hyperparameters
H = 30  #network size for each layer
H2 = 30
H3 = 20
batch_num = 1000 #want to lears some things, but not all - balance against learning rate
learn_rate = 5e-4 #1/2000 learning rate
gamma = 0.9995 #how much do we weight short term rewards vs long term reqards
q_copy_count = 2000 #how many learn events do we do before copying the active q net to the q prime
explore_w = .99994 # how quickly we stop exploring the network and use experiences instead
min_explore = 0.03 #minimum level that we let explore_w reach
max_episodes = 2000 #number of attempts allowed
max_steps = 500 #max number of steps allowed in an attempt
mem_size = 100000 #determines how much state memory we can maintain (state, reward, new state, terminal)



#create envoronment and setup recording, force clears existing training data
env = gym.make('CartPole-v1')
env.monitor.start('/tmp/training_dir', force=True)
tf.reset_default_graph()


#set up network variables
#first layer shape depends on the state/observation inputs
#final layer shape depends on the action space available
#intermediate layers depend on hyperparameters
w1 = tf.Variable(tf.random_uniform([env.observation_space.shape[0], H], -1.0, 1.0))
b1 = tf.Variable(tf.random_uniform([H], -1.0, 1.0))

w2 = tf.Variable(tf.random_uniform([H, H2], -1.0, 1.0))
b2 = tf.Variable(tf.random_uniform([H2], -1.0, 1.0))

w3 = tf.Variable(tf.random_uniform([H2, H3], -1.0, 1.0))
b3 = tf.Variable(tf.random_uniform([H3], -1.0, 1.0))

w4 = tf.Variable(tf.random_uniform([H3, env.action_space.n], -1.0, 1.0))
b4 = tf.Variable(tf.random_uniform([env.action_space.n], -1.0, 1.0))

#create q' net
w1p = tf.Variable(tf.random_uniform([env.observation_space.shape[0], H], -1.0, 1.0))
b1p = tf.Variable(tf.random_uniform([H], -1.0, 1.0))

w2p = tf.Variable(tf.random_uniform([H, H2], -1.0, 1.0))
b2p = tf.Variable(tf.random_uniform([H2], -1.0, 1.0))

w3p = tf.Variable(tf.random_uniform([H2, H3], -1.0, 1.0))
b3p = tf.Variable(tf.random_uniform([H3], -1.0, 1.0))

w4p = tf.Variable(tf.random_uniform([H3, env.action_space.n], -1.0, 1.0))
b4p = tf.Variable(tf.random_uniform([env.action_space.n], -1.0, 1.0))

#create pointers so that when we run a session we can direct the Q prime to update from the Q net
#update_q = [w1p.assign(w1), w2p.assign(w2), w3p.assign(w3), w4p.assign(w4), b1p.assign(b1), b2p.assign(b2), b3p.assign(b3), b1p.assign(b4)]
w1p_up = w1p.assign(w1)
b1p_up = b1p.assign(b1)
w2p_up = w2p.assign(w2)
b2p_up = b2p.assign(b2)
w3p_up = w3p.assign(w3)
b3p_up = b3p.assign(b3)
w4p_up = w4p.assign(w4)
b4p_up = b4p.assign(b4)
update_q = [w1p_up, w2p_up, w3p_up, w4p_up, b1p_up, b2p_up, b3p_up, b4p_up]



#build 4 layer neural network with relu
prev_states = tf.placeholder(tf.float32, [None, env.observation_space.shape[0]])
hidden_1 = tf.nn.relu(tf.matmul(prev_states, w1) + b1)
hidden_2 = tf.nn.relu(tf.matmul(hidden_1, w2) + b2)
hidden_25= tf.nn.dropout(hidden_2, .75)  #add some noise in the middle of the net
hidden_3 = tf.nn.relu(tf.matmul(hidden_25, w3) + b3)
Q_net = tf.matmul(hidden_3, w4) + b4   #removed squeeze b/c i removed Q prime squeeze



hidden_1p = tf.nn.relu(tf.matmul(prev_states, w1p) + b1p)
hidden_2p = tf.nn.relu(tf.matmul(hidden_1p, w2p) + b2p)
hidden_3p = tf.nn.relu(tf.matmul(hidden_2p, w3p) + b3p)
Qp_net = tf.matmul(hidden_3p, w4p) + b4p     #removed squeeze b/c that was killing max below


# used in training -come back and figure this shit out

# env.action_space.n are the actions available for this scenario (left right up down etc.)
# action used is the list of actions already taken, when one hot is used it takes the action index number and turns it into an array [wtf does it actually mean]
actions_used = tf.placeholder(tf.int32, [None])
action_masks = tf.one_hot(actions_used, env.action_space.n)  

#i think filtered q is the list of actions taken previously with all the actions we didnt take removed 
#  take the action masks (which indicate the actions taken up to now?) so:
#     which consisted of [0 1; 0 1; 1 0; 0 1...] where each line in the array indicated an action (right; right; left; right)
# and elementwise multiply with the Q network states whose result is then narrowed down to a single dimension
# this is then the list of previous values earned from prev actions that can be used for calculating the loss??? maybe? dont quite know why still
filtered_Q = tf.reduce_sum(tf.mul(Q_net, action_masks), reduction_indices=1) 

#training
# A list of expected rewards with each elementrepresenting the reward on a specific step [next action, 2nd action reward, 3ed action reward...]
# loss calculated as the MSE of the previous reward(???) against the expected reward(???)
#train with adam optimizer to minimize the MSE(?)
Expected_reward_q = tf.placeholder(tf.float32, [None,])  
loss = tf.reduce_sum(tf.square(filtered_Q - Expected_reward_q))
train = tf.train.AdamOptimizer(learn_rate).minimize(loss)

#env

#running explore probability, reduces at the (explore_w) rate
explore = 1.0

# D is an array that contains the history up to the memory limit (mem_size)
# Each element in the list is made up of [starting state, action, reward, ending state, done]
D = []

# Running total of reward earned in the current episode
reward_total = 0

# variables used for state plotting
xmax = 1
ymax = 1
xind = 1
yind = 3


with tf.Session() as sess:
    # initialize session and set initial Q prime weight
    sess.run(tf.initialize_all_variables())
    sess.run(update_q)
    
    # counter for the number of steps taken since the last q copy was made
    q_step_count = 0
    
    #episode attempt loop; xrange slightly faster than range
    for episode in xrange(max_episodes):
        #for each episode:
        # initialize state as we reset env
        # reward total initialized to 0
        # done - false until the env reports that the episode has ended
        state = env.reset()
        reward_total = 0.0
        done = False

        #step (observation/action) loop
        for step in xrange(max_steps):
            # increment step counter for replacing q prime
            q_step_count += 1
            
            # update state variables for graph
            xmax = max(xmax, state[xind])
            ymax = max(ymax, state[yind])
            
            #every 10 episodes we want to observe the attempt and print the current Q and Q prime
            if episode % 10 == 0:
                q, qp = sess.run([Q_net,Qp_net], feed_dict={prev_states: np.array([state])})
                #print "Q:{}, Q_ {}".format(q[0], qp[0])
                #print "T: {} S {}".format(q_step_count, state)
                env.render()
            
            # if the running exploration probability is greater than a random value then we set action to a random choice
            # if it is less, then we use the network to pick the action
            if explore > random.random():
                action = env.action_space.sample()
            else:
                q = sess.run(Q_net, feed_dict={prev_states: np.array([state])})[0]    #not sure why we need [0] at the end
                action = np.argmax(q)
                #print action
            
            # update the running exploration probability but no less than the minimum
            explore = max(explore * explore_w, min_explore)
            
            #now that we have a chosen action, take a step and record results
            new_state, reward, done, info = env.step(action)
            
            # tack on reward for this step to running reward total
            reward_total += reward
            
            #print results episode results when done
            if done:
                print"{} - {} - {}".format(episode, step, explore)
            
            #Save everything we just discovered into memory up to the memory limit, if we hit it drop the oldest memory
            D.append([state, action, reward, new_state, done])
            if len(D) > mem_size:
                D.pop(0); 
            
            #now that we saved everything about the step we just took, update the current state
            state = new_state
            
            #if we failed or the episode ended break the step loop
            if done: break
            
            #pull a random set of data from our memory to use in training the network
            samples = random.sample(D, min(batch_num, len(D)))
#            print samples
            
            #Extracts the new_state item list from sample data set and calculate future Q values based on the memory sample
            new_states = [ x[3] for x in samples]
            #For each state in the new_state list calculate values based on Q prime network(?) do it here instead of within for loop (batch in instead of single)
            all_q_prime = sess.run(Qp_net, feed_dict={prev_states: new_states})
            
            #create variables for looking forward and evaluating states against q prime
            #y_ future reward guess, state_samples states from run, actions - list of actions to take, terminal - terminal counter
            y_ = []
            state_samples = []
            actions = []
            terminalcount = 0
            
            # step through each sample pulled from memory; 
            for index, sample_i in enumerate(samples):
                #assign the elements of the sample to temporary cycling variables
                state_mem, curr_action, reward, new_state, done = sample_i
                
                # if this sample indicates we're done then increment terminal counter
                if done:
                    terminalcount += 1
                    
                #extract the q prime we're interested in from the batch run above and get the max result
                maxq = max(all_q_prime[index])
                
                #tack on the reward received plus the weighted future reward - assuming we aren't terminal
                y_.append(reward + (gamma * maxq * (not done)))
                
                #tack on the state to the sample state list and the action to the action list
                state_samples.append(state_mem)
                actions.append(curr_action)
            #end of for loop    
            
            #run training session based on the data pulled from the sample memory
            sess.run([train], feed_dict={prev_states: state_samples, Expected_reward_q: y_, actions_used: actions})
            
            
            #every x steps we want to overwrite our q prime network with q
            if q_step_count % q_copy_count == 0:
                sess.run(update_q)
                print "Q Prime updated"
            
        # end of step loop

        #every x episodes we want to see graph of how q is doing... nothing for now
#        if episode % 100 == 0:
             
            
    #end of episode loop
    #close environment
env.monitor.close()





