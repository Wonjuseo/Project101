# Basic tensorflow examples
# The sources were from Tensorflow for Deep learning reasearch - lecture
import tensorflow as tf
# 3 + 5 = 8
a = tf.add(3,5)
# Tensor("Add:0", shape=(), dtype=int32)
# It was not our wanted value.
print(a)
# To get the value, we added the code "sess = tf.Session()"
sess = tf.Session()
# To see the result, we wrote the code "print(sess.run(a))"
print(sess.run(a))
# The result was 8
# More complex calculation
x = 2
y = 3
# op1 is adding, op2 is multipling and op3 is powing.
op1 =tf.add(x,y)
op2 =tf.multiply(x,y)
op3 =tf.pow(op2,op1)
# The results was 7776
print(sess.run(op3))
# The Session is closed.
sess.close()
# Another way to declear the session
with tf.Session() as sess:
    # It showed the same result.
    print(sess.run(op3))
    # To calculate op3, op2 and op1, we added the code like below
    print(sess.run([op3, op2, op1]))
    k,y,z = sess.run([op3,op2,op1])
    # It showed the same result
    print(k,y,z)
    # Creating a graph:
