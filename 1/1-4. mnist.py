# tensorflow를 사용할 것이기에 import 합니다.
import tensorflow as tf

# MNIST data를 받아옵니다. MNIST data는 0~9까지의 숫자들 입니다.
# 본 Code에서는 MNIST data를 학습하고 새로운 입력(숫자)를 판단해내는 방법을 설명합니다.

from tensorflow.examples.tutorials.mnist import input_data

# one-hot encoding을 합니다. 예를들어 숫자가 1이라면 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] 이 됩니다.
# 숫자가 2라면 [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] 이 됩니다.
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)


# Step 1 Neural network
# 숫자 이미지는 28 x 28 픽셀로 이루어져 있어 총 784개의 데이터를 가집니다.
X = tf.placeholder(tf.float32)
# 결과는 0~9의 10 가지 경우입니다.
Y = tf.placeholder(tf.float32)

# (1 x 784) x (784 x 300) x (300 x 512) x (512 x 1024) x (1024 x 10) = (1 x 10) 
# 4 layer의 가중치를 선언해줍니다.
W1 = tf.Variable(tf.random_normal([784,300],stddev = 0.01))
W2 = tf.Variable(tf.random_normal([300,512],stddev = 0.01))
W3 = tf.Variable(tf.random_normal([512,1024],stddev = 0.01))
W4 = tf.Variable(tf.random_normal([1024,10],stddev = 0.01))
# 각 Layer의 연산을 보여줍니다. 
L1 = tf.nn.sigmoid(tf.matmul(X,W1))
L2 = tf.nn.sigmoid(tf.matmul(L1,W2))
L3 = tf.nn.sigmoid(tf.matmul(L2,W3))
hypothesis = tf.matmul(L3,W4)
# Cost function으로는 soft max function을 사용합니다.
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= hypothesis,labels= Y))
# 이전 GradientDescent와 다른 AdamOptimizer를 사용합니다.
train = tf.train.AdamOptimizer(0.001).minimize(cost)

# Session을 선언합니다.
with tf.Session() as sess:
    # 변수를 초기화 해줍니다.
    init = tf.global_variables_initializer()
    sess.run(init)
    batch_size = 100
    total_batch = int(mnist.train.num_examples/batch_size)

    # 총 20번의 학습을 진행합니다.
    for step in range(20):
        # cost 값을 저장해주는 변수를 추가합니다.
        sum_cost = 0
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict={X: batch_xs, Y: batch_ys})
            sum_cost += sess.run(cost, feed_dict={X: batch_xs, Y: batch_ys})

        print("Step:",step,"Average cost:",sum_cost/total_batch)

    print("Optimization Finished")
    # 학습 이후의 모델이 얼마나 잘 예측을 하는지 확인합니다.
    pred = tf.equal(tf.argmax(hypothesis,1),tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(pred,tf.float32))
    # mnist.test.images를 입력으로 넣어서 판단한 것이 실제 숫자와 얼마나 같은지를 나타냅니다.
    # 1에 해당하는 Image를 넣었을 때에 1로 판단을 하면 이 경우에는 Accuracy가 100이 됩니다.
    # 이를 전체 mnist.test.images를 넣어서 실제 숫자와 같은지를 확인하고 Accuracy를 보여줍니다.
    # 저는 97.8%가 나왔습니다.
    print("Accuracy:",sess.run(accuracy,feed_dict={X:mnist.test.images, Y:mnist.test.labels}))
