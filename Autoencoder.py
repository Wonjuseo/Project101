# 본 Code에서 사용할 tensorflow, matplotlib.pyplot, nupmy, random을 import한다.
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random

# MNIST data를 불러오고 이를 one_hot encoding합니다.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

# parameter 설정
learning_rate = 0.01
training_epoch = 15
batch_size = 100

#Hidden layer의 Feature 개수
n_hidden = 300

# 입력의 크기 28 x 28 pixels
n_input = 28*28

# Step 1 Neural network setting
# Y는 placeholder로 선언되지 않습니다.
X = tf.placeholder(tf.float32, [None, n_input])

# input -> encoder -> decoder -> output
# Encoder는 정보를 압축하여 Feature를 얻어냅니다.
W1 = tf.Variable(tf.random_normal([n_input,n_hidden]))
B1 = tf.Variable(tf.random_normal([n_hidden]))
# Deocder는 출력을 입력값과 동일하게 하여 입력과 같은 아웃풋을 만들어 냅니다.
W2 = tf.Variable(tf.random_normal([n_hidden,n_input]))
B2 = tf.Variable(tf.random_normal([n_input]))

encoder = tf.nn.sigmoid(tf.add(tf.matmul(X,W1),B1))
decoder = tf.nn.sigmoid(tf.add(tf.matmul(encoder,W2),B2))

# Decoder는 입력과 비슷한 결과를 내야합니다.
Y = X
# 입력과 비슷하게 Decoder의 출력이 나와야 하기 때문에 Cost function으로 decoder와 실제 값의 차이의 제곱으로 정합니다.
# Cost function의 값이 크다면 실제 값과 Decoding된 결과가 다르다는 것을 의미합니다.
cost = tf.reduce_mean(tf.pow(Y - decoder,2))
train = tf.train.AdamOptimizer(learning_rate).minimize(cost)

total_batch = int(mnist.train.num_examples/batch_size)

# Step 2 Training
with tf.Session() as sess:
    init = tf.global_variables_initializer()

    sess.run(init)

    for epoch in range(training_epoch):
        sum_cost = 0

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict={X:batch_xs})
            sum_cost += sess.run(cost,feed_dict={X:batch_xs})

        print("Epoch:",epoch,"Avg Cost:",sum_cost/total_batch)
    
    print("Optimization finished")

    # Decoding

    pred = sess.run(decoder,feed_dict={X:mnist.test.images[:10]})
    figure, axis = plt.subplots(2,10,figsize=(10,2))

    for i in range(10):
        axis[0][i].set_axis_off()
        axis[1][i].set_axis_off()
        axis[0][i].imshow(np.reshape(mnist.test.images[i],(28,28)))
        axis[1][i].imshow(np.reshape(pred[i],(28,28)))

    plt.show()


