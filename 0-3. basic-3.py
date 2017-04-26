import tensorflow as tf
# Matrix and scalar
a = tf.constant([1, 2])
b = tf.constant([3, 5])
# add, multiply and matmul
op1 = tf.add(a,b)
op2 = tf.multiply(a,b)
op3 = tf.matmul(tf.reshape(a,[1,2]),tf.reshape(b,[2,1]))
# Variable W
W = tf.Variable(10)
assign_op = W.assign(100)
# In order to use W variable, we should initialized it
init = tf.global_variables_initializer()
# Placeholder
# Can assemble the graph first without knowing the values needed for computation
p = tf.placeholder(tf.float32)
p2 = tf.placeholder(tf.float32)
# Add
addp = p+p2
# Multiply
mulp = p*p2

with tf.Session() as sess:
    #Initialize the variable
    sess.run(init)
    r1, r2, r3 = sess.run([op1,op2,op3])

    print(r1,r2,r3)
    # r1 = [4 7] 1+3 = 4 2+5 = 7
    # r2 = [3 10] 1x3 = 3  2x5 = 10
    # r3 = [[13]] 1x3+2x5= 13
    # the same results
    print(op1.eval(),op2.eval(),op3.eval())
    # op1.eval() = [4 7] 1+3 = 4 2+5 = 7
    # op2.eval() = [3 10] 1x3 = 3  2x5 = 10
    # op3.eval() = [[13]] 1x3+2x5= 13
    print(sess.run(W)) # 10
    print(sess.run(assign_op)) # 100
    print(sess.run(W)) # 100
    # Use placeholder p =3 and p2 =2 
    # the result were 5 and 6
    print(sess.run([addp, mulp],feed_dict={p:3.,p2:2.}))
