import tensorflow as tf
import numpy as np

xy = np.loadtxt('train.txt', unpack=True, dtype='float32')

x_data = xy[0: -1]
y_data = xy[-1]

print('x', x_data)
print('y', y_data)

W = tf.Variable(tf.random_uniform([1, 3], -1.0, 1.0))

hypothesis = tf.matmul(W, x_data)

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

# step 1: node 선택
cost_hist = tf.summary.scalar("cost_scalar", cost)

# step 2: summary 통합.
merged = tf.summary.merge_all()

a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

# step 3: writer 생성
summary_writer = tf.summary.FileWriter("../tensorboard", graph=tf.get_default_graph())

for step in range(2001):
    _, summary = sess.run([train, merged])
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W))
        summary_writer.add_summary(summary, step)
