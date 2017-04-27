import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr
import tensorflow as tf

def rargmax(vector):
    # random argmax
    m = np.max(vector)
    indices = np.nonzero(vector == m)[0]
    return pr.choice(indices)

# Reward Update Q
# Algorithm
# For each s,a initialize table entry Q(s,a)<-0
# Observe current stat s
# Do foever:
# select an action a and execute it
# receive immediate reward
# observe the new state
# update the table entry for Q(s,a)
# update the state

# Non-deterministic environment
env = gym.make('FrozenLake-v0')
# Intialize table with all zeros
Q = np.zeros([env.observation_space.n,env.action_space.n])
# Discount factor
dis = .99
# Learning_rate
learning_rate = 0.85
# Set learning parameters
num_episodes = 2000

# Create lists to contain total rewards and steps per episode
rList = []

for i in range(num_episodes):
    # Reset environment and get first new observation
    # Intialize state
    state = env.reset()
    rAll = 0
    done = False
    # Decaying E-greedy
    e = 1. /((i/100)+1)
    while not done:
        # Determine actions
        # Exploit and Exploration : Decaying E-greedy
        if pr.random() < e:
            action = env.action_space.sample()
        else:
            action = rargmax(Q[state,:])
        # Add Random noise
        # action = np.argmax(Q[state,:]+np.random.randn(1,env.action_space.n)/(i+1))
        # Get new state and reward from environment
        new_state, reward, done, _ = env.step(action)
        # Update Q-table with new knowledge using decay rate and learning rate
        Q[state,action] = (1-learning_rate)*Q[state,action] +learning_rate*(reward + dis * np.max(Q[new_state,:]))
        # Update the state
        state = new_state
        # reward every episode
        rAll += reward

    rList.append(rAll)
# Show the result
print("Success rate:"+str(sum(rList)/num_episodes))
# Show the table
print(Q)
plt.bar(range(len(rList)),rList,color="blue")
plt.show()