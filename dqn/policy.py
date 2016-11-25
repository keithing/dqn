import tempfile

import numpy as np
import tensorflow as tf


def weight_var(shape, name="weights"):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, name=name)


def bias_var(shape, name="bias"):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial, name=name)


def conv_layer(X, weight_shape, stride, scope):
    """Convolutional layer"""
    with tf.variable_scope(scope):
        w = weight_var(weight_shape)
        b = bias_var([weight_shape[-1]])
        c = tf.nn.conv2d(X, w, strides=[1, stride, stride, 1], padding='VALID')
        h = tf.nn.relu(c + b)
    return h


def fc_layer(X, weight_shape, bias_shape, relu, scope):
    """Fully connected layer"""
    with tf.variable_scope(scope):
        w = weight_var(weight_shape)
        b = bias_var(bias_shape)
        linear_output = tf.matmul(X, w) + b
        h = tf.nn.relu(linear_output) if relu else linear_output
    return h


# TODO
# Increasing the Action Gap: New Operators for Reinforcement Learning
# Adds an aditional penalty to gradient using target network to
# predict current s as well as s prime. Psuedocode:
#
# v = target_network.predict(s)
# q = target_network.predict(s_prime) + reward
# alpha = .1
# err = (q_hat - q) - alpha * (q_hat - v)
# sse = reduce_sum(err ** 2)
# active_learning_train_step = Optimizer(epsilon).minimize(sse)

class DQNPolicy:

    def __init__(self, window=4, warm_start=None):
        self.window = window
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.build_tf_graph(window)
            self.session = tf.Session()
            self.target_session = tf.Session()
            self.saver = tf.train.Saver()
        if warm_start:
            self.saver.restore(self.session, warm_start)
        else:
            self.session.run(self._init)
        self.update_target_network()

    def fit(self, memory, n_updates, batchsize, gamma=0.99, verbose=True):
        errs = []
        for _ in range(n_updates):
            s, a, q = self._minibatch(memory, batchsize, gamma=gamma)
            _, err = self.session.run(
                [self._train_step, self._sse],
                feed_dict = {self._s: s, self._a: a, self._q: q})
            errs.append(err)
        if verbose:
            err_mean = np.mean(errs)
            err_se = np.std(errs)
            msg = "Error: {} +/- {}".format(err_mean, err_se)
            print(msg, flush=True)

    def predict(self, data):
        return self.session.run(self._q_hats, feed_dict = {self._s: data})

    def checkpoint(self, path):
        self.saver.save(self.session, path)

    def update_target_network(self):
        """ Refresh target network weights to current model's weights."""
        with tempfile.NamedTemporaryFile() as f:
            self.checkpoint(f.name)
            self.saver.restore(self.target_session, f.name)

    def close_sessions():
        self.session.close()
        self.target_session.close()

    def _minibatch(self, memory, batchsize, gamma):
        samples = memory.sample(batchsize, self.window)
        s, a, q = [], [], []
        for sample in samples:
            s_prime = [np.swapaxes(sample["s_prime"], 0, 2)]
            target_qs = self.target_session.run(q_hats,
                                                feed_dict = {self._s: s_prime})
            s.append(np.swapaxes(sample["s"], 0, 2))
            a.append(sample["action"])
            q.append(sample["reward"] + np.max(target_qs) * gamma)
        return s, self._linear_index(a, 6, batchsize), q

    def _linear_index(self, x, i, batchsize):
        """
        Workaround for tensorflow.  Cannot slice on last dimension of tensor,
        so we need to reshape matrix into vector and use linear indexing.
        http://stackoverflow.com/questions/36764791/in-tensorflow-how-to-use-tf-gather-for-the-last-dimension
        """
        return x + (np.array(range(batchsize)) * i)

    def build_tf_graph(self, window):

        # Input Variables
        with tf.variable_scope("input"):
            self._a = tf.placeholder(tf.int32, shape=[None], name="a")
            self._s = tf.placeholder(tf.float32, shape=[None, 84, 84, window],
                                     name="s")
            self._q = tf.placeholder(tf.float32, shape=[None], name="q")

        # Convolutional layers
        self._h_0 = conv_layer(self._s, [8, 8, window, 32], 4, "h_0")
        self._h_1 = conv_layer(self._h_0, [4, 4, 32, 64], 2, "h_1")
        self._h_2 = conv_layer(self._h_1, [3, 3, 64, 64], 1, "h_2")
        with tf.variable_scope("h2_flat"):
            self._h_2_flat = tf.reshape(self._h_2, [-1, 7 * 7 * 64])

        # Fully connected layers
        self._h_3 = fc_layer(self._h_2_flat, [7 * 7 * 64, 512], [512],
                             True, "h_3")
        self._q_hats = fc_layer(self._h_3, [512, 6], [6], False, "output")

        # Training
        with tf.variable_scope("train"):
            self._q_hat = tf.gather(tf.reshape(self._q_hats, [-1]), self._a)
            self._squared_errors = tf.squared_difference(self._q_hat, self._q)
            clip_err = tf.clip_by_value(self._squared_errors, 0, 1)
            self._sse = tf.reduce_sum(clip_err)
            opt = tf.train.AdamOptimizer(0.00001)
            self._train_step = opt.minimize(self._sse)

        # Initialization
        with tf.variable_scope("init"):
            self._init = tf.initialize_all_variables()

class DDQNPolicy(DQNPolicy):

    def _minibatch(self, memory, batchsize, gamma):
        samples = memory.sample(batchsize)
        s, a, q = [], [], []
        for sample in samples:
            s_prime = [np.swapaxes(sample["s_prime"], 0, 2)]
            online_qs = self.session.run(self._q_hats,
                                         feed_dict = {self._s: s_prime})[0]
            tar_qs = self.target_session.run(self._q_hats,
                                             feed_dict = {self._s: s_prime})[0]
            s.append(np.swapaxes(sample["s"], 0, 2))
            a.append(sample["action"])
            q.append(sample["reward"] + gamma * tar_qs[np.argmax(online_qs)])
        return s, self._linear_index(a, 6, batchsize), q
