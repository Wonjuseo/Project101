# GAN: Generative Adversarial Network

# tensorflow, matplotlib.pyplot, numpy를 import합니다.
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# MNIST data를 불러오고, one-hot encoding을 합니다.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

# Parameter setting
total_epoch = 30
batch_size =100
learning_rate = 0.0002

# Neural network features
n_hidden = 256

# The size of inputs
n_input = 28 * 28

# Generator의 입력으로 사용될 Noise의 크기
n_noise = 128

# Step 1 Neural network
X = tf.placeholder(tf.float32,[None,n_input])
Z = tf.placeholder(tf.float32,[None,n_noise])

# Generator variables
W1 = tf.Variable(tf.random_normal([n_noise,n_hidden],stddev= 0.01))
B1 = tf.Variable(tf.zeros([n_hidden]))
W2 = tf.Variable(tf.random_normal([n_hidden,n_input],stddev = 0.01))
B2 = tf.Variable(tf.zeros([n_input]))

# Discriminator

W3 = tf.Variable(tf.random_normal([n_input,n_hidden],stddev =0.01))
B3 = tf.Variable(tf.zeros([n_hidden]))

W4 = tf.Variable(tf.random_normal([n_hidden,1],stddev= 0.01))
B4 = tf.Variable(tf.zeros([1]))

# Generator neural network
def generator(noise):
    hidden_layer = tf.nn.relu(tf.matmul(noise,W1)+B1)
    generated_outputs = tf.sigmoid(tf.matmul(hidden_layer,W2)+B2)
    return generated_outputs

def discriminator(inputs):
    hidden_layer = tf.nn.relu(tf.matmul(inputs,W3)+B3)
    discrimination = tf.sigmoid(tf.matmul(hidden_layer,W4)+B4)
    return discrimination

def gen_noise(batch_size):
    return np.random.normal(size=[batch_size,n_noise])

# Generate random image
G = generator(Z)

# Get the value by using a image from noise
D_gene = discriminator(G)

# Get the value by using a real image
D_real = discriminator(X)

# Optimization: Maximize loss_G and loss_D
# To maximize loss_D, minimize D_gene
loss_D = tf.reduce_mean(tf.log(D_real)+tf.log(1-D_gene))

loss_G = tf.reduce_mean(tf.log(D_gene))

G_var_list = [W1,B1,W2,B2]
D_var_list = [W3,B3,W4,B4]

train_D = tf.train.AdamOptimizer(learning_rate).minimize(-loss_D, var_list=D_var_list)
train_G = tf.train.AdamOptimizer(learning_rate).minimize(-loss_G, var_list=G_var_list)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    total_batch = int(mnist.train.num_examples/batch_size)
    loss_val_D,loss_val_G = 0, 0

    for epoch in range(total_epoch):

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            noise = gen_noise(batch_size)

            _, loss_val_D = sess.run([train_D,loss_D],feed_dict={X:batch_xs,Z:noise})
            _, loss_val_G = sess.run([train_G,loss_G],feed_dict={Z:noise})

        print("Epoch:",epoch,"D loss:",loss_val_D,"G loss:",loss_val_G)

        k_noise = gen_noise(10)

        pred = sess.run(G,feed_dict={Z:k_noise})

        figure, axis = plt.subplots(1, 10, figsize=(10,1))

        for i in range(10):
            axis[i].set_axis_off()
            axis[i].imshow(np.reshape(pred[i],(28,28)))

        plt.savefig("./samples/{}.png".format(str(epoch).zfill(3)), bbox_inches='tight')
        plt.close(figure)

    print("Optimization finished")