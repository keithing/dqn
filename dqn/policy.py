import tempfile

import numpy as np
import tensorflow as tf

###############################################################################
# Layer Creation Helper Functions
###############################################################################


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


###############################################################################
# Define Graph
###############################################################################


# Input Variables
with tf.variable_scope("input"):
    a = tf.placeholder(tf.int32, shape=[None], name="a")
    s = tf.placeholder(tf.float32, shape=[None, 84, 84, 4], name="s")
    q = tf.placeholder(tf.float32, shape=[None], name="q")

# Convolutional layers
h_0 = conv_layer(s, [8, 8, 4, 32], 4, "h_0")
h_1 = conv_layer(h_0, [4, 4, 32, 64], 2, "h_1")
h_2 = conv_layer(h_1, [3, 3, 64, 64], 1, "h_2")
with tf.variable_scope("h2_flat"):
    h_2_flat = tf.reshape(h_2, [-1, 7 * 7 * 64])

# Fully connected layers
h_3 = fc_layer(h_2_flat, [7 * 7 * 64, 512], [512], True, "h_3")
q_hats = fc_layer(h_3, [512, 6], [6], False, "output")

# Training
with tf.variable_scope("train"):
    q_hat = tf.gather(tf.reshape(q_hats, [-1]), a)
    squared_errors = tf.squared_difference(q_hat, q)
    sse = tf.reduce_sum(tf.clip_by_value(squared_errors, 0, 1))
    train_step = tf.train.AdamOptimizer(0.00001).minimize(sse)

# Initialization
with tf.variable_scope("init"):
    init = tf.initialize_all_variables()

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


###############################################################################
# Deep Q Networks
###############################################################################


class DQNPolicy:

    def __init__(self, warm_start=None):
        self.session = tf.Session()
        self.target_session = tf.Session()
        self.saver = tf.train.Saver()
        if warm_start:
            self.saver.restore(self.session, warm_start)
        else:
            self.session.run(init)
        self.update_target_network()

    def fit(self, memory, n_updates, batchsize, gamma=0.99, verbose=True):
        errs = []
        for _ in range(n_updates):
            _s, _a, _q = self._minibatch(memory, batchsize, gamma=gamma)
            _, err = self.session.run(
                [train_step, sse],
                feed_dict = {s: _s, a: _a, q: _q})
            errs.append(err)
        if verbose:
            err_mean = np.mean(errs)
            err_se = np.std(errs)
            msg = "Error: {} +/- {}".format(err_mean, err_se)
            print(msg, flush=True)

    def predict(self, data):
        return self.session.run(q_hats, feed_dict = {s: data})

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
        samples = memory.sample(batchsize)
        _s, _a, _q = [], [], []
        for sample in samples:
            s_prime = [np.swapaxes(sample["s_prime"], 0, 2)]
            target_qs = self.target_session.run(q_hats,
                                                feed_dict = {s: s_prime})
            _s.append(np.swapaxes(sample["s"], 0, 2))
            _a.append(sample["action"])
            _q.append(sample["reward"] + np.max(target_qs) * gamma)
        return _s, self._linear_index(_a, 6, batchsize), _q

    def _linear_index(self, x, i, batchsize):
        """ 
        Workaround for tensorflow.  Cannot slice on last dimension of tensor,
        so we need to reshape matrix into vector and use linear indexing.
        http://stackoverflow.com/questions/36764791/in-tensorflow-how-to-use-tf-gather-for-the-last-dimension
        """ 
        return x + (np.array(range(batchsize)) * i)


class DDQNPolicy(DQNPolicy):

    def _minibatch(self, memory, batchsize, gamma):
        samples = memory.sample(batchsize)
        _s, _a, _q = [], [], []
        for sample in samples:
            s_prime = [np.swapaxes(sample["s_prime"], 0, 2)]
            online_qs = self.session.run(q_hats,
                                         feed_dict = {s: s_prime})[0]
            target_qs = self.target_session.run(q_hats, 
                                                feed_dict = {s: s_prime})[0]
            _s.append(np.swapaxes(sample["s"], 0, 2))
            _a.append(sample["action"])
            _q.append(sample["reward"] +
                      gamma * target_qs[np.argmax(online_qs)])
        return _s, self._linear_index(_a, 6, batchsize), _q
