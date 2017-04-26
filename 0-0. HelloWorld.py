
import tensorflow as tf

hello = tf.constant('Hello, World!')
# Start tensorflow session
with tf.Session() as sess:
    # Run the operation
    print(sess.run(hello))
