import tensorflow as tf

x=[[2,3]]
y=[[2,4]]
op1 = tf.add(x,y)
op2 = x+y
op3 = tf.pow(op1,op2)

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs',sess.graph)
    r = sess.run(op3)
    print(r)
    