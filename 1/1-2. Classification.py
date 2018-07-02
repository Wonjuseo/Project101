# tensorflow를 사용하기 위해서 tensorflow를 import 합니다.
# 배열을 사용할때 편한 tool로 numpy를 import합니다.
import tensorflow as tf
import numpy as np

# Step 1
# Train시킬 데이터를 만듭니다.
# 본 Code에서는 XOR(Exclusive -OR)을 만들어보도록 하겠습니다.
x_data = np.array([[0,0],[0,1],[1,0],[1,1]])
y_data = np.array([[0],[1],[1],[0]])

# Step 2
# Data를 저장할 수 있는 Placeholder를 선언합니다.

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Step 3
# Neural network(신경망)을 만들기 위해서 2개의 Layer를 형성합니다.
# W1의 크기는 2 x 30 인데 2는 입력의 수입니다. x_data는 각 row에 2개의 데이터를 가집니다.
W1 = tf.Variable(tf.random_uniform([2,30],-1.0,1.0))
W2 = tf.Variable(tf.random_uniform([30,1],-1.0,1.0))

# Bias를 설정해줍니다. W1과 x의 연산 결과 30개의 데이터가 나오므로 b1의 크기는 30이 됩니다.
# (1 x 2) x (2 x 30) = (1 x 30) 이 되고 W2의 크기가 (30 x 1)이 되어 연산결과 1개의 데이터가 나오므로 b2의 크기는 1이 됩니다. 
b1 = tf.Variable(tf.zeros([30]))
b2 = tf.Variable(tf.zeros([1]))

# Classification은 Linear regression과 달리 어떤 특정 값이 True 혹은 False로 되는 경우이기에 0~1 사이의 값을 나타내기 위해서
# Sigmoid function을 사용합니다.
# 두 층의 Layer를 만듭니다. 1 layer로는 XOR을 해결할 수가 없습니다.
L1 = tf.sigmoid(tf.add(tf.matmul(X,W1),b1))
L2 = tf.sigmoid(tf.add(tf.matmul(L1,W2),b2))
hypothesis = L2

# Cost function으로 Cross_entropy 함수를 사용합니다.
# hypothesis가 0, Y가 0인 경우에 cost가 0이 되고, hypothesis 가 1, Y가 0인 경우에 cost는 무한대가 됩니다.
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
# GradientDescent algorithm을 사용하여 Global minimum을 찾아 데이터를 학습합니다.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Session을 선언해줍니다.
with tf.Session() as sess:
    # 위에서 선언된 변수들을 초기화 해줍니다.
    init = tf.global_variables_initializer()
    sess.run(init)
    # 총 100001번의 학습을 합니다.
    for step in range(100001):
        sess.run(train,feed_dict={X:x_data,Y:y_data})
        # Step이 10000번이 될 때마다 현재의 cost값을 보여줍니다.
        if step % 10000 == 0:
            print(step, sess.run(cost, feed_dict={X:x_data,Y:y_data}))

    # 실제로 학습된 모델이 데이터가 들어왔을때에 잘 예측하는지 확인합니다.
    pred = tf.floor(hypothesis+0.5)
    real = Y
    # x_data를 hypothesis에 적용시켰을 때의 결과와 Real data를 비교합니다.
    print("Prediction:",sess.run(pred,feed_dict={X:x_data}))
    print("Real:",sess.run(real,feed_dict={Y:y_data}))

