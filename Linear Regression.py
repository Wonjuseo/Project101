import tensorflow as tf

# Step 1
# Data를 만들어 놓고 이를 가지고 linear regression을 할 것입니다.
x_data = [1, 2, 3, 4, 5, 6, 7]
y_data = [1, 2, 3, 4, 5, 6, 7]
# W,b는 linear regression에서의 각각 기울기, 절편을 의미합니다. 따라서 변수로 선언을 하여 변할 수 있게합니다.
W = tf.Variable(tf.zeros([1]))
b = tf.Variable(tf.zeros([1]))
# X,Y는 각각 x_data, y_data를 받아들입니다.
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
# linear regression에서의 hypothesis(가정)은 y = W * x + b입니다.
hypothesis = tf.add(tf.multiply(X,W),b)
# cost function을 선언합니다. 우리가 가정한 식과 실제 식이 얼마나 다른지를 나타냅니다.
# 따라서, 학습을 하는 것은 이 cost function의 값을 줄이도록 앞에서 선언한 변수 W, b를 수정하는게 됩니다.
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# Linear regression에서의 cost function은 볼록함수의 형태를 가지고 있습니다.
# 볼록함수의 Global Minimum을 찾기위해선 tensorflow에서는 GradientDescent Algorithm을 제공합니다.
# learning_rate는 쉽게 설명하면 어느정도로 움직일까에 관련된 변수입니다.
# learning_rate이 크면 빠르게 학습을 하지만 발산할 위험이 있으며, 작으면 발산할 위험은 줄어들지만 느리게 학습을 합니다.
# 따라서 여러번 실험을 통해서 자신의 모델에 적합한 learning_rate를 찾는 것이 중요합니다.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
# optimizer는 cost를 줄이는 방향으로 학습을 합니다.
train = optimizer.minimize(cost)

# Tensorflow를 동작시키기 위해서 Session을 선언합니다.
with tf.Session() as sess:
    # 위에서 선언해준 변수들은 초기화를 해주어야합니다. 초기화가 되지 않으면 동작하지 않으므로 밑의 코드로 초기화를 합니다.
    init = tf.global_variables_initializer()
    sess.run(init)
    # 학습을 2000회정도 실행합니다.
    for step in range(2000):
        # 앞에서 선언된 X,Y 에 Data를 입력하여 학습을 시행합니다.
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        # 100 step마다 현재의 cost function의 값과 변수 W, b값을 보여줍니다.
        if step % 100 == 0:
            print(step, sess.run(cost,feed_dict={X:x_data,Y:y_data}), sess.run(W), sess.run(b))


    # 학습이 종료되었음을 알려주고 새로운 값에 대한 결과를 보여줍니다.
    print("Training Finished")
    print("X: 10, Y:", sess.run(hypothesis, feed_dict={X: 10}))
    print("X: 13, Y:", sess.run(hypothesis, feed_dict={X: 13}))