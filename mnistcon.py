import os

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


def w_var(shape):
    init = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(init)


def b_var(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)


def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1],padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')

learning_rate = 0.01
batch_size = 128
n_epochs = 10
mnist = input_data.read_data_sets('/data/mnist', one_hot=True)
X = tf.placeholder(tf.float32, [batch_size, 784], name="image_holder")
Y = tf.placeholder(tf.float32, [batch_size, 10], name="label_holder")

W_conv1 = w_var([5,5,1,32])
b_conv1 = b_var([32])
x_image = tf.reshape(X, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = w_var([5, 5, 32, 64])
b_conv2 = b_var([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


W_fc1 = w_var([7 * 7 * 64, 1024])
b_fc1 = b_var([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = w_var([1024, 10])
b_fc2 = b_var([10])

logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y, name="loss")
loss = tf.reduce_mean(entropy)

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./log_graph', graph=sess.graph)
    start_time = time.time()
    sess.run(tf.global_variables_initializer())
    n_batches = int(mnist.train.num_examples/batch_size)
    for i in range(n_epochs): # train the model n_epochs times
        total_loss = 0
        for _ in range(n_batches):
            X_batch, Y_batch = mnist.train.next_batch(batch_size)
            # TO-DO: run optimizer + fetch loss_batch
            _, loss_batch = sess.run([optimizer, loss], feed_dict={X: X_batch, Y:Y_batch, keep_prob:0.6})
            # print(total_loss,"+",loss_batch)
            total_loss += loss_batch
        print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))

    print('Total time: {0} seconds'.format(time.time() - start_time))

    print('Optimization Finished!')

