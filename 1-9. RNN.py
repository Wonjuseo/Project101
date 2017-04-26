# RNN(Recurrent Neual Network) - LSTM

import tensorflow as tf
from tensorflow.contrib import rnn

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# MNIST image shape is 28 x 28 pixel
# 28 sequences of 28 steps for every sample

# Parameter
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 28 # MNIST data input (image: 28 x 28)
n_steps = 28 # Steps
n_hidden = 256 # Hidden layer of fuatures
n_classes = 10 # 0~9

# Placeholder
x = tf.placeholder(tf.float32,[None,n_steps,n_input])
y = tf.placeholder(tf.float32,[None,n_classes])

# Weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes],stddev = 0.01))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes],stddev = 0.01))
}

def RNN(x, weights, biases):

    x = tf.unstack(x, n_steps, 1)
    # Define a lstm cell
    lstem_cell = rnn.BasicLSTMCell(n_hidden,forget_bias = 1.0)

    outputs, states = rnn.static_rnn(lstem_cell,x,dtype=tf.float32)

    return tf.matmul(outputs[-1],weights['out'])+biases['out']

pred = RNN(x,weights,biases)
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
corret = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(corret,tf.float32))

# Initialize the variables
init = tf.global_variables_initializer()

# Start the session
with tf.Session() as sess:
    sess.run(init)
    step = 1

    while step*batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 sequence of 28 elemets
        batch_x = batch_x.reshape((batch_size,n_steps,n_input))
        # Train
        sess.run(optimizer, feed_dict={x:batch_x,y:batch_y})
        if step % display_step == 0:
            acc = sess.run(accuracy,feed_dict={x:batch_x, y:batch_y})
            loss = sess.run(cost,feed_dict={x:batch_x,y:batch_y})
            print("Iter:",str(step*batch_size),"Minibatch loss:",loss,"Acc:",acc)

        step += 1
    
    print("Optimization Finished")

    # Predict new data
    test_len = 200
    test_data = mnist.test.images[:test_len].reshape((-1,n_steps,n_input))
    test_label = mnist.test.labels[:test_len]
    print("Accuracy:",sess.run(accuracy,feed_dict={x:test_data,y:test_label}))
