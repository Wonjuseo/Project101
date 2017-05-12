# Character Sequence Softmax

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

# Set the seed, reproducibility
tf.set_random_seed(1234) 
# Long string
sentence = "It's my life, It's now or never. I ain't gonna live forever. I just want to live while i'm alive. It's my life."
# index -> char
char_set = list(set(sentence))
# char -> index
char_dic = {w: i for i, w in enumerate(char_set)}

# Hyperparameters
data_dim = len(char_set)
hidden_size = len(char_set)
output_size = len(char_set)
# Arbitrary number
sequence_length = 10

dataX = []
dataY = []

for i in range(0,len(sentence)-sequence_length):
    x_str = sentence[i:i+sequence_length]
    y_str = sentence[i+1:i+sequence_length+1]
    print(i,x_str,'->',y_str)

    # x,y string -> index
    x = [char_dic[c] for c in x_str]
    y = [char_dic[c] for c in y_str]

    dataX.append(x)
    dataY.append(y)

batch_size = len(dataX)

# Placeholders
X = tf.placeholder(tf.int32, [None,sequence_length])
Y = tf.placeholder(tf.int32, [None,sequence_length])

# One hot encoding
X_one_hot = tf.one_hot(X,output_size)
X_for_softmax = tf.reshape(X_one_hot,[-1,hidden_size])
# Make a lstem cell with hidden_size
cell = rnn.BasicLSTMCell(hidden_size,state_is_tuple=True)
cell = rnn.MultiRNNCell([cell]*5,state_is_tuple=True)
# outputs
outputs, _states = tf.nn.dynamic_rnn(cell,X_one_hot,dtype = tf.float32)

# Fully connected layer
X_for_fc = tf.reshape(outputs,[-1,hidden_size])
outputs = tf.contrib.layers.fully_connected(X_for_fc,output_size,activation_fn =None)
# Reshape output for seqnece_loss
outputs = tf.reshape(outputs,[batch_size,sequence_length,output_size])
# All weights are 1 (equal weights)
weights = tf.ones([batch_size,sequence_length])

# Loss and train
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs,targets=Y,weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
# Store the total sentence
string2 = []
# Session
with tf.Session() as sess:
    # Initialize the tensorflow variables
    sess.run(tf.global_variables_initializer())
    # Learning
    for i in range(500):
        _,l, results = sess.run([train, loss, outputs],feed_dict={X:dataX,Y:dataY})
        for j, result in enumerate(results):
            index = np.argmax(result,axis=1)
            print(i,j,''.join([char_set[t] for t in index]),l)
    # Print the last char of each result
    results = sess.run(outputs,feed_dict={X:dataX})
    for j, result in enumerate(results):
        index = np.argmax(result,axis=1)
        if j is 0:
            string2.append(''.join([char_set[t] for t in index]))
        else:
            string2.append(char_set[index[-1]])

    print(string2)
