# Cart Pole example
# Deep Q- networks
# Deep - Go deep
# Minibatch - Replay memory and train minibatch

# Algorithm
# Build Networks and initialize them
# Environments
# E-greedy and action , get the reward
# No train save data on the buffer
# random sample and train

import random
import numpy as np
import tensorflow as tf
# Please put in same file dqn.
import dqn
import gym
from collections import deque

# Environment
env = gym.make('CartPole-v0')
# Discount factor
dis = 0.9
rList = []
REPLAY_MEMORY = 50000

# Parameters
input_size = env.observation_space.shape[0]
output_size = env.action_space.n

def simple_replay_train(DQN, train_batch):
    x_stack = np.empty(0).reshape(0,DQN.input_size)
    y_stack = np.empty(0).reshape(0,DQN.output_size)

    # Get stored information from the buffer
    for state, action, reward, next_state, done in train_batch:
        Q = DQN.predict(state)

        # terminal
        if done:
            Q[0,action] =reward
        else:
            Q[0,action] = reward + dis*np.max(DQN.predict(next_state))
        y_stack = np.vstack([y_stack,Q])
        x_stack = np.vstack([x_stack, state])

    return DQN.update(x_stack,y_stack)

def bot_play(mainDQN):
    # See our trained newtork in action
    k = 0
    s = env.reset()
    reward_sum = 0
    while True:
        env.render()
        a = np.argmax(mainDQN.predict(s))
        s, reward, done, _ = env.step(a)
        reward_sum += reward
        if done:
            print("Total score:",reward_sum)
            s = env.reset()
            k += 1
            reward_sum = 0
            if k>10:                
                break
def main():
    max_episodes = 1000

    replay_buffer = deque()
    with tf.Session() as sess:
        mainDQN = dqn.DQN(sess,input_size,output_size)
        # Initialize variables
        init = tf.global_variables_initializer()
        sess.run(init)

        for i in range(max_episodes):
            # e- greedy
            e = 1./((i/10)+1)
            step_count = 0
            state = env.reset()
            done = False

            # Before fail
            while not done:
                # E-greedy
                if random.gauss(0,1) < e:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(mainDQN.predict(state))
                # Get the observation results
                next_state, reward, done, _ = env.step(action)            
                # After fail
                if done:
                    if step_count >= 199:
                        reward = 200
                    else:
                        reward = -100
                
                replay_buffer.append((state,action,reward,next_state,done))
                if len(replay_buffer) >REPLAY_MEMORY:
                    replay_buffer.popleft()
                # Update state
                state = next_state
                step_count += 1
                if step_count > 199:
                    break

            print("Episode:",i,"step:",step_count)
            if step_count> 10000:
                pass
        
            # train every 10 episodes
            if i % 10 == 1: 
                # Get a random batch of experience
                for _ in range(50):
                    # Minibatch works better
                    minibatch = random.sample(replay_buffer,10)
                    loss, _ = simple_replay_train(mainDQN,minibatch)

                print("Loss:",loss)
        # Rendering
        bot_play(mainDQN)

# python start code
if __name__ == "__main__":
    main()
