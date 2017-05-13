# RNN - basic
# hihello

import tensorflow as tf
import numpy as np

# Set the seed
# Reproducibility
tf.set_random_seed(1234)

# Total character data
idx2char = ['h','i','e','l','o']

# Input: hihell -> output: ihello
x_data = [[0,1,0,2,3,3]] #hi,hell
x_one_hot = [[[1, 0, 0, 0, 0],   # h 0
              [0, 1, 0, 0, 0],   # i 1
              [1, 0, 0, 0, 0],   # h 0
              [0, 0, 1, 0, 0],   # e 2
              [0, 0, 0, 1, 0],   # l 3
              [0, 0, 0, 1, 0]]]  # l 3

y_data = [[1, 0, 2, 3, 3, 4]]    # ihello

num_classes = 5
input_dim =5 # one-hot size
hidden_size = 5 # output from the LSTM
batch_size = 1 # one sentence
sequence_length = 6 # the length of input data
learning_rate = 0.1
# Placeholders
X = tf.placeholder(tf.float32,
[None,sequence_length,input_dim]) # Input
Y = tf.placeholder(tf.int32,[None,sequence_length]) # Label
# RNN - LSTM Cell
cell = tf.contrib.rnn.BasicLSTMCell(num_units = hidden_size,
state_is_tuple = True)
initial_state = cell.zero_state(batch_size,tf.float32)
outputs, _ = tf.nn.dynamic_rnn(
    cell,X,initial_state=initial_state,dtype = tf.float32
)

# Fully connected layer
X_for_fc = tf.reshape(outputs,[-1,hidden_size])
outputs = tf.contrib.layers.fully_connected(
    inputs=X_for_fc,num_outputs=num_classes, activation_fn = None
)

# Reshape out for sequence_loss

outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

weights = tf.ones([batch_size,sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits = outputs, targets=Y,weights = weights
)
# Training
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
# Prediction
prediction = tf.argmax(outputs,axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(50):
        l, _ =sess.run([loss, train], feed_dict={X:x_one_hot,Y:y_data})
        result = sess.run(prediction, feed_dict={X:x_one_hot})
        print(i,"loss:",l,"prediction:",result,"true Y:",y_data)

        # print char using dic

        result_str = [idx2char[c] for c in np.squeeze(result)]
        print("\tprediction str:",''.join(result_str))
