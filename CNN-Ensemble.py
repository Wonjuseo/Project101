# MNIST adn deep learning CNN
# Class, Layer and Ensemble

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

# Set the seed of program
tf.set_random_seed(1234)
# Read data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

# Hyper parameters
learning_rate = 0.001
training_epochs = 20
batch_size = 100

class Model:
    def __init__(self,sess,name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            
            # Testing or training
            self.training = tf.placeholder(tf.bool)

            self.X = tf.placeholder(tf.float32,[None,784])

            # Image size and labels
            X_img = tf.reshape(self.X,[-1,28,28,1])
            self.Y = tf.placeholder(tf.float32, [None, 10])

            # Convolutional layer 1
            conv1 = tf.layers.conv2d(inputs=X_img,
            filters = 32, kernel_size=[3,3],
            padding = "SAME",activation=tf.nn.relu)

            # Pooling layer 1
            pool1 = tf.layers.max_pooling2d(inputs=conv1,
            pool_size=[2,2],padding="SAME",strides=2)

            dropout1 = tf.layers.dropout(inputs=pool1,
            rate = 0.7, training = self.training)

            # Convolutional layer 2 and pooling layer 2
            conv2 = tf.layers.conv2d(inputs=dropout1,
            filters = 64, kernel_size=[3,3],
            padding = "SAME",activation=tf.nn.relu)

            pool2 = tf.layers.max_pooling2d(inputs=conv2,
            pool_size=[2,2],padding="SAME",strides=2)

            dropout2 = tf.layers.dropout(inputs=pool2,
            rate = 0.7, training = self.training)

            # Convolutional layer 2 and pooling layer 2
            conv3 = tf.layers.conv2d(inputs=dropout2,
            filters = 128, kernel_size=[3,3],
            padding = "SAME",activation=tf.nn.relu)

            pool3 = tf.layers.max_pooling2d(inputs=conv3,
            pool_size=[2,2],padding="SAME",strides=2)

            dropout3 = tf.layers.dropout(inputs=pool3,
            rate = 0.7, training = self.training)

            # Dense layer with relu

            flat = tf.reshape(dropout3,[-1,128*4*4])
            dense4 = tf.layers.dense(inputs=flat,units=625,
            activation=tf.nn.relu)
            dropout4 = tf.layers.dropout(inputs=dense4,
            rate = 0.5, training = self.training)

            # Logits layer : L5 final FC 625 inputs
            self.logits = tf.layers.dense(inputs=dropout4,units=10)

        # Define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels= self.Y
        ))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate = learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(
            tf.argmax(self.logits,1),tf.argmax(self.Y,1)
        )
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    def predict(self,x_test,training=False):
        return self.sess.run(self.logits,
        feed_dict={self.X:x_test,self.training:training})

    def get_accuracy(self,x_test,y_test,training=False):
        return self.sess.run(self.accuracy,
        feed_dict={self.X:x_test,self.Y:y_test,self.training:training})

    def train(self,x_data,y_data,training=True):
        return self.sess.run([self.cost, self.optimizer],
        feed_dict={self.X: x_data, self.Y: y_data, self.training: training})
# Initialize the session
sess = tf.Session()
# Models
models = []
num_models = 2
for m in range(num_models):
    models.append(Model(sess,"model"+str(m)))

sess.run(tf.global_variables_initializer())

# Train my models
for epoch in range(training_epochs):
    avg_cost_list = np.zeros(len(models))
    total_batch = int(mnist.train.num_examples/batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        # Train
        for m_idx, m in enumerate(models):
            c, _ = m.train(batch_xs,batch_ys)
            avg_cost_list[m_idx] += c/total_batch

    print("Epoch:",epoch+1," cost:",avg_cost_list)

print("Learning finished")

# Test the model and calculate accuracy

test_size = len(mnist.test.labels)
predictions = np.zeros(test_size*10).reshape(test_size,10)
for m_idx,m in enumerate(models):
    print(m_idx,"accuracy:",m.get_accuracy(
        mnist.test.images, mnist.test.labels
    ))
    p = m.predict(mnist.test.images)
    predictions += p

ensemble_correct_prediction = tf.equal(tf.argmax(predictions,1),
tf.argmax(mnist.test.labels,1))
ensemble_accuracy = tf.reduce_mean(
    tf.cast(ensemble_correct_prediction,tf.float32)
)
print("Ensemble accuracy:",sess.run(ensemble_accuracy))
