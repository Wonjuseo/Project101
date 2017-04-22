# tensorflow, numpy를 사용하기위해 import

import tensorflow as tf
import numpy as np

# Deep learning을 위해 데이터를 읽습니다.
data = np.loadtxt('./data.csv',delimiter=',',unpack=True,dtype='float32')

# csv자료의 0부터 2번째 까지의 feature를 x_data에 넣습니다.
# 나머지는 분류가 되는 데이터로 y_data에 넣습니다.
x_data = np.transpose(data[0:3])
y_data = np.transpose(data[3:])

# Step 1 Neural network
# x_data, y_data를 저장하기위해 X, Y를 placeholder형태로 선언해줍니다.
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# 3층의 weights를 선언합니다. (1 x 3) x (3 x 50) x (50 x 100) x (100 x 3) = (1 x 3)
W1 = tf.Variable(tf.random_uniform([3,50],-1.,1.))
W2 = tf.Variable(tf.random_uniform([50,100],-1.,1.))
W3 = tf.Variable(tf.random_uniform([100,3],-1.,1.))

# 3층의 layer를 선언합니다. Classification 경우로 0~1 값이 나오도록 Sigmoid 함수를 사용합니다.
L1 = tf.sigmoid(tf.matmul(X,W1))
L2 = tf.sigmoid(tf.matmul(L1,W2))
L3 = tf.matmul(L2,W3)

# 3개의 점수 A,B,C 중에서 하나를 고르는 것이므로 cost function을 softmax_cross_entropy 함수로 정한다.
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=L3,labels= Y))
# 0.001 learning_rate로 Global minimum을 찾는다.
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(cost)

# Session을 선언한다.
with tf.Session() as sess:
    # 변수들을 초기화 한다.
    init = tf.global_variables_initializer()
    sess.run(init)
    # 총 20001번을 Train 한다.
    for step in range(20001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        # step이 1000번이 될때마다 현재 cost를 보여준다.
        if step % 1000 == 0:
            print (step, sess.run(cost, feed_dict={X: x_data, Y: y_data}))
        
    # 학습 이후에 데이터를 넣었을 때에 결과가 어떻게 나오는지를 확인한다.
    pred = tf.argmax(L3,1)
    real = tf.argmax(Y,1)

    print("Prediction:",sess.run(pred,feed_dict={X:x_data}))
    print("Real:",sess.run(real, feed_dict={Y:y_data}))

    #세 과목에서 80, 80, 80점을 받았을 때의 결과를 보여준다. 
    print("Grade: ",sess.run(pred,feed_dict={X:[[80,80,80]]}))








