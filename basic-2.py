import tensorflow as tf
# a and b are constants that are not be changed
# More readable, we put in the name
a = tf.constant(2,name="a")
b = tf.constant(3,name="b")
# x is adding a and b
x = tf.add(a,b,name="add")
# The matrix can be constant
c = tf.constant([2,2],name="c")
d = tf.constant([[0,1],[2,3]],name="d")
# y is adding c and d, but the sizes of c and d are not same
# What happen? 
y = tf.add(c,d,name="matadd")
# tf.zeros_like creates a tensor of shape and type as the input_tensor but all elements are zeros
z = tf.zeros_like(c)
# tf.ones_like did like tf.zeros_like but all elements are ones
q = tf.ones_like(c)
# tf.fill create a tensor filled with a sclar value
k = tf.fill([2,3],5)

with tf.Session() as sess:
    # This showed 5
    print(sess.run(x))
    # This showed [2 3] [4 5].
    # [2,2] were added in both rows.
    print(sess.run(y))
    # z = [0 0]
    print(sess.run(z))
    # q = [1 1]
    print(sess.run(q))
    # k = [5 5 5] [5 5 5]
    print(sess.run(k))
    # tf.linspace
    # 1 3.25 5.5 7.75 10
    print(sess.run(tf.linspace(1.0,10.0,5)))
    # tf.range
    # 1 4 7
    print(sess.run(tf.range(1.0,10.0,3)))
    # Random_shuffle
    # 3.25 1 10 5.5 7.75
    print(sess.run(tf.random_shuffle(tf.linspace(1.0,10.0,5).eval())))
    
