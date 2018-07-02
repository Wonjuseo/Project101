import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)
lr = 0.0005
epochs = 10
batch_size = 100
display_step = 1

def layer(input,weight_shape,bias_shape):
    weight_stddev = (2.0/weight_shape[0])**0.5
    w_init = tf.random_normal_initializer(stddev=weight_stddev)
    bias_init = tf.constant_initializer(value=0)
    W = tf.get_variable("W",weight_shape,initializer=w_init)
    b = tf.get_variable("b",bias_shape,initializer=bias_init)
    return tf.nn.relu(tf.matmul(input,W)+b)

def loss(output,y):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=y)
    loss = tf.reduce_mean(xentropy)
    return loss

def training(cost,global_step):
    tf.summary.scalar("cost",cost)
    optimizer = tf.train.AdamOptimizer(lr)
    train_op = optimizer.minimize(cost,global_step=global_step)
    return train_op

def evaluate(output,y):
    correct_prediction = tf.equal(tf.argmax(output,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    return accuracy


def conv2d(input,weight_shape,bias_shape):
    in_ = weight_shape[0] * weight_shape[1] * weight_shape[2]
    weight_init = tf.random_normal_initializer(stddev=(2.0/in_)**0.5)

    W = tf.get_variable("W",weight_shape,initializer=weight_init)
    bias_init = tf.constant_initializer(value=0)
    b = tf.get_variable("b",bias_shape,initializer=bias_init)
    conv_out = tf.nn.conv2d(input,W,strides=[1,1,1,1],padding="SAME")
    return tf.nn.relu(tf.nn.bias_add(conv_out,b))

def max_pool(input,k=2):
    return tf.nn.max_pool(input,ksize=[1,k,k,1],strides=[1,k,k,1],padding="SAME")

def inference(x,keep_prob=0.5):
    x = tf.reshape(x,shape=[-1,28,28,1])
    with tf.variable_scope("conv_1"):
        conv_1 = conv2d(x,[5,5,1,32],[32])
        pool_1 = max_pool(conv_1)
    
    with tf.variable_scope("conv_2"):
        conv_2 = conv2d(pool_1,[5,5,32,64],[64])
        pool_2 = max_pool(conv_2)
    
    with tf.variable_scope("fc"):
        pool_2_flat = tf.reshape(pool_2,[-1,7*7*64])
        fc_1 = layer(pool_2_flat,[7*7*64,1024],[1024])
        # drop out
        fc_1_drop = tf.nn.dropout(fc_1,keep_prob)
    
    with tf.variable_scope("output"):
        output = layer(fc_1_drop,[1024,10],[10])
    return output

with tf.Graph().as_default():
    x = tf.placeholder("float",[None,784])
    y = tf.placeholder("float",[None,10])

    output = inference(x)
    cost = loss(output,y)
    global_step = tf.Variable(0,name='global_step',trainable=False)
    train_op = training(cost,global_step)
    eval_op = evaluate(output,y)
    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()
    sess = tf.Session()
    summary_writer = tf.summary.FileWriter("logistic_logs/",graph_def=sess.graph_def)

    init_op = tf.initialize_all_variables()
    sess.run(init_op)

    for epoch in range(epochs):
        avg_cost =0
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            mbatch_x, mbatch_y = mnist.train.next_batch(batch_size)
            feed_dict = {x:mbatch_x,y:mbatch_y}
            sess.run(train_op,feed_dict=feed_dict)
            minibatch_cost = sess.run(cost,feed_dict=feed_dict)
            avg_cost += minibatch_cost/total_batch
        if epoch % display_step == 0:
            val_feed_dict ={x: mnist.validation.images,
            y:mnist.validation.labels
            }
            accuracy = sess.run(eval_op,feed_dict=val_feed_dict)
            print("Validation error : ",(1-accuracy))
            summary_str = sess.run(summary_op,feed_dict=feed_dict)
            summary_writer.add_summary(summary_str,sess.run(global_step))
            saver.save(sess,"logistic_logs/model-checkpoint",global_step = global_step)
    print("Finished")

