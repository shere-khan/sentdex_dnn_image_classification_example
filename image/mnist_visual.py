import tensorflow as tf
import datetime
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])

W = tf.get_variable("weights1", shape=[784, 10],
                    initializer=tf.glorot_uniform_initializer())

b = tf.get_variable("bias1", shape=[10],
                    initializer=tf.constant_initializer(0.1))

y = tf.nn.relu(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])

# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
# train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
tf.summary.scalar("loss", cost)
train_step = tf.train.AdamOptimizer().minimize(cost)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()

tf.summary.scalar('accuracy', accuracy)

time_string = datetime.datetime.now().isoformat()
experiment_name = f"one_hidden_layer_1000_steps_{time_string}"

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(f'logs/train/{experiment_name}', sess.graph)
test_writer  = tf.summary.FileWriter(f'logs/test/{experiment_name}', sess.graph)

tf.global_variables_initializer().run()

for step in range(10000):
    print(f"training step: {step}")
    if step % 10 == 0:
        summary, acc = sess.run([merged, accuracy], feed_dict={x: mnist.test.images,
                                                               y_: mnist.test.labels})
        test_writer.add_summary(summary, step)
        print(f'Step {step}; Model accuracy: {acc}')
    else:
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})
        summary = sess.run(merged, feed_dict={x: batch_xs,
                                              y_: batch_ys})
        train_writer.add_summary(summary, step)