# Character Sequence Softmax

import tensorflow as tf
import numpy as np

# Set the seed, reproducibility
tf.set_random_seed(1234) 

sentence = "South Korea"
# index -> char
idx2char = list(set(sentence))
# char -> index
char2idx = {c: i for i, c in enumerate(idx2char)}

# Hyperparameters
input_size = len(char2idx)
rnn_hidden_size = len(char2idx)
output_size = len(char2idx)
# one sentence 
batch_size = 1
sequence_length = len(sentence) -1

# char to index
sample_idx = [char2idx[c] for c in sentence]
# x_data
x_data = [sample_idx[:-1]]
# y_data - labels
y_data = [sample_idx[1:]]

# Placeholders
X = tf.placeholder(tf.int32, [None,sequence_length])
Y = tf.placeholder(tf.int32, [None,sequence_length])

# One hot encoding
X_one_hot = tf.one_hot(X,output_size)
X_for_softmax = tf.reshape(X_one_hot,[-1,rnn_hidden_size])

# Softmax layer
softmax_w = tf.get_variable("softmax_w", [rnn_hidden_size, output_size])
softmax_b = tf.get_variable("softmax_b",[output_size])

# Hypothesis
outputs = tf.matmul(X_for_softmax,softmax_w)+softmax_b

outputs = tf.reshape(outputs,[batch_size,sequence_length,output_size])
weights = tf.ones([batch_size,sequence_length])

# Loss and train
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs,targets=Y,weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

# Predict the label of data
prediction = tf.argmax(outputs,axis=2)

# Session
with tf.Session() as sess:
    # Initialize the tensorflow variables
    sess.run(tf.global_variables_initializer())
    # Learning
    for i in range(2000):
        l, _ = sess.run([loss, train],feed_dict={X:x_data,Y:y_data})
        result = sess.run(prediction, feed_dict={X:x_data})

        result_str  = [idx2char[c] for c in np.squeeze(result)]
        # Result: 'outh Kouea'
        print(i,"loss:",l,"prediction:",''.join(result_str))

