# Step 0 setting

import tensorflow as tf

# Use Data and one-hot encoding
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

# Step 1 Neural network
# Use ReLU instead of sigmoid
# 28->14->7-> Fully connected
# Use dropout to solve over-fitting problems.

X = tf.placeholder(tf.float32,[None,28,28,1])
Y = tf.placeholder(tf.float32,[None,10])
drop_out = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_normal([3,3,1,20],stddev = 0.01))

L1 = tf.nn.relu(tf.nn.conv2d(X,W1,strides=[1,1,1,1],padding='SAME'))
L1 = tf.nn.max_pool(L1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
L1 = tf.nn.dropout(L1,drop_out)

W2 = tf.Variable(tf.random_normal([3,3,20,40],stddev = 0.01))
L2 = tf.nn.relu(tf.nn.conv2d(L1,W2,strides=[1,1,1,1],padding='SAME'))
L2 = tf.nn.max_pool(L2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
L2 = tf.reshape(L2,[-1,7*7*40])
L2 = tf.nn.dropout(L2,drop_out)

W3 = tf.Variable(tf.random_normal([7*7*40,256],stddev = 0.01))
L3 = tf.nn.relu(tf.matmul(L2,W3))
L3 = tf.nn.dropout(L3,drop_out)

W4 = tf.Variable(tf.random_normal([256,10],stddev=0.01))
hypothesis = tf.matmul(L3,W4)
# Step 2 training part
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= hypothesis, labels=Y))
# learning_rate = 0.001
train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
# Start tf.Session
with tf.Session() as sess:
    #initialize our variables
    init = tf.global_variables_initializer()
    sess.run(init)

    batch_size = 100
    total_batch = int(mnist.train.num_examples/batch_size)

    # Training part
    for step in range(15):
        sum_cost = 0

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            batch_xs = batch_xs.reshape(-1, 28, 28, 1)
            sess.run(train, feed_dict={X: batch_xs, Y: batch_ys,drop_out: 0.8})
            cost2 = sess.run(cost, feed_dict={X:batch_xs,Y:batch_ys,drop_out: 0.8})
            sum_cost += cost2

        print("Step:",step,"Avg Cost:",sum_cost/total_batch)
        
    # Step 3 prediction
    pred = tf.equal(tf.argmax(hypothesis,1),tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(pred,tf.float32))
    # accuracy 99.1%
    print("Accuracy:",sess.run(accuracy,feed_dict={X:mnist.test.images.reshape(-1, 28, 28, 1), Y:mnist.test.labels,drop_out:1.0}))
