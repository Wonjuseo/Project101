# Cart Pole example
# Q- networks

import random
import numpy as np
import tensorflow as tf
import gym

# Environment
env = gym.make('CartPole-v0')
num_episodes = 500
dis = 0.99
rList = []

# Parameters
input_size = env.observation_space.shape[0]
output_size = env.action_space.n
learning_rate = 0.1
# Placeholder
X = tf.placeholder(tf.float32,[None,input_size])
# Weights
W1 = tf.Variable(tf.random_normal(shape=[input_size,output_size], stddev = 0.01))
Qpred = tf.matmul(X,W1)
Y = tf.placeholder(shape=[None,output_size],dtype = tf.float32)
# Cost function
loss = tf.reduce_sum(tf.square(Y-Qpred))
# Training
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Initialize variables
init = tf.global_variables_initializer()
with tf.Session() as sess:   
    # Initialize variables
    sess.run(init)
    # Start learning
    for i in range(num_episodes):
        # e- greedy
        e = 1./((i/10)+1)
        rAll = 0
        step_count = 0
        s = env.reset()
        done = False

        # Before fail
        while not done:
            step_count += 1
            x = np.reshape(s,[1,input_size])
            # Predict next action
            Qs = sess.run(Qpred,feed_dict={X:x})
            # E-greedy
            if random.random() < e:
                a = env.action_space.sample()
            else:
                a = np.argmax(Qs)
            # Get the observation results
            s1, reward, done, _ = env.step(a)
            
            # After fail
            if done:
                # Fail
                Qs[0,a] = -100
            else:
                # Continuously working
                x1 = np.reshape(s1,[1,input_size])
                # Predict next state action
                Qs1 = sess.run(Qpred, feed_dict={X:x1})
                # Up date
                Qs[0,a] = reward + dis * np.max(Qs1)
            # Training
            sess.run(train,feed_dict={X:x,Y:Qs})
            # Update state
            s = s1
        
        rList.append(step_count)
        print("Episode:",i,"step:",step_count)
        if len(rList)>10 and np.mean(rList[-10:])>500:
            break
    # Reset
    observation = env.reset()
    reward_sum = 0
    while True:
        # Rendering
        env.render()

        x = np.reshape(observation, [1, input_size])
        Qs = sess.run(Qpred,feed_dict={X:x})
        a = np.argmax(Qs)

        observation, reward, done, _ = env.step(a)
        reward_sum += reward
        if done:
            print("Total score:",reward_sum)
            break
