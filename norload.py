import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf 

######################################## 
## NORMAL LOADING   			      ##
## print out a graph with 1 Add node  ## 
########################################

x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')
z = tf.add(x, y)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	writer = tf.summary.FileWriter('./graphs/l2', sess.graph)
	for _ in range(10):
		sess.run(z)
	print(tf.get_default_graph().as_graph_def())
	writer.close()
