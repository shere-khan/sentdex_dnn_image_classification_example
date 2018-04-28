import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neural_network_model(data):
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {
        'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
        'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {
        'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
        'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes]))}

    # (input_data * weights) + biases

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output

def train_nn(x):
    model = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=y))
    # tf.summary.histogram("cross entropy hist", cost)
    tf.summary.scalar("cross entropy scal", cost)

    # learning_rate = (default) 0.001
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # cycles feed forward + back prop
    hm_epochs = 2

    with tf.Session() as sess:
        # Writers
        train_writer = tf.summary.FileWriter('logs/train')
        test_writer = tf.summary.FileWriter('logs/test')

        # Train acc
        correct = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
        test_accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy: {0}'.format(
            test_accuracy.eval({x: mnist.test.images, y: mnist.test.labels})))
        tf.summary.scalar("accuracy", test_accuracy)
        sess.run(tf.initialize_all_variables())

        counter = 0
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                counter += 1
                merge = tf.summary.merge_all()
                if _ % 10 == 0:
                    summary, acc = sess.run([merge, test_accuracy],
                                            feed_dict={x: mnist.test.images,
                                                       y: mnist.test.labels})
                    train_writer.add_summary(summary, counter)
                else:
                    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                    _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})
                    summary = sess.run(merge, feed_dict={x: batch_xs.images,
                                                         y_: batch_ys})
                    train_writer.add_summary(summary, _)

                epoch_loss += c

            # Training accuracy
            # correct = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
            # training_accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            # print('Accuracy: {0}'.format(
            #     training_accuracy.eval({x: mnist.train.images, y: mnist.train.labels})))

            # Test accuracy
            # correct = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
            # test_accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            # print('Accuracy: {0}'.format(
            #     test_accuracy.eval({x: mnist.test.images, y: mnist.test.labels})))

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss: ', epoch_loss)
        train_writer.close()

train_nn(x)
