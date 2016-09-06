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
    a = tf.placeholder(tf.int32, name="a")
    s = tf.placeholder(tf.float32, shape=[None, 84, 84, 4], name="s")
    r = tf.placeholder(tf.float32, shape=[None], name="r")
    q_prime = tf.placeholder(tf.float32, shape=[None, 6], name="q_prime")
    gamma = tf.placeholder(tf.float32, name="gamma")

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
    q = tf.add(r, tf.mul(gamma, tf.reduce_max(q_prime, 1)))
    q_hat = tf.gather(tf.reshape(q_hats, [-1]), a)
    squared_errors = tf.squared_difference(q_hat, q)
    sse = tf.reduce_sum(tf.clip_by_value(squared_errors, 0, 1))
    train_step = tf.train.AdamOptimizer(0.00001).minimize(sse)

# Initialization
with tf.variable_scope("init"):
    init = tf.initialize_all_variables()


##############################################################################
# Deep Q Learning Model
##############################################################################

class DQN:

    def __init__(self, batchsize=32, warm_start=None):
        self.batchsize = batchsize
        self.session = tf.Session()
        self.target_session = tf.Session()
        self.saver = tf.train.Saver()
        if warm_start:
            self.saver.restore(self.session, warm_start)
        else:
            self.session.run(init)
        self.update_target_network()

    def fit(self, memory, n_updates=5, verbose=True):
        errs = []
        for _ in range(n_updates):
            s_, a_, r_, q_prime_ = self.gen_minibatch(memory)
            _, err, x, y = self.session.run(
                [train_step, sse,q, q_hat],
                feed_dict = {s: s_, a: a_, r: r_,
                             q_prime: q_prime_, gamma: .99})
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
        with tempfile.NamedTemporaryFile() as f:
            self.checkpoint(f.name)
            self.saver.restore(self.target_session, f.name)

    def gen_minibatch(self, memory):
        samples = memory.sample(self.batchsize)
        s_, q_prime_, a_, r_ = [], [], [], []
        for sample in samples:
            s_.append(np.swapaxes(sample["s"], 0, 2))
            a_.append(sample["action"])
            r_.append(sample["reward"])
            s_prime = [np.swapaxes(sample["s_prime"], 0, 2)]
            q_hats_ = self.target_session.run(q_hats,  feed_dict = {s: s_prime})
            q_prime_.append(q_hats_)
        return s_, self.linear_index(a_, 6), np.squeeze(r_), np.squeeze(q_prime_)

    def linear_index(self, x, i):
        """ 
        Workaround for tensflow.  Cannot slice on last dimension of tensor,
        so we need to reshape matrix into vector and use linear indexing.
        http://stackoverflow.com/questions/36764791/in-tensorflow-how-to-use-tf-gather-for-the-last-dimension
        """ 
        return x + (np.array(range(self.batchsize)) * i)

    def close_sessions():
        self.session.close()
        self.target_session.close()
