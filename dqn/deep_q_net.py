from copy import copy
import os 
import argparse
import numpy as np
import tensorflow as tf

###############################################################################
# Layer Creation Helper Functions
###############################################################################

def weight_var(shape, name="weights"):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=name)


def bias_var(shape, name="bias"):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=name)


def c_layer(X, weight_shape, stride, scope):
    """Convolutional layer"""
    with tf.variable_scope(scope):
        w = weight_var(weight_shape)
        b = bias_var([weight_shape[-1]])
        c = tf.nn.conv2d(X, w, strides=[1, stride, stride, 1], padding='VALID')
        h = tf.nn.relu(c + b)
    return h


def f_layer(X, weight_shape, bias_shape, relu, scope):
    """Fully connected (flat) layer"""
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
    r = tf.placeholder(tf.float32, shape=[None, 6], name="r")
    q_prime = tf.placeholder(tf.float32, shape=[None, 6], name="q_prime")
    gamma = tf.placeholder(tf.float32, name="gamma")

# Convolutional layers
h_0 = c_layer(s, [8, 8, 4, 32], 4, "h_0")
h_1 = c_layer(h_0, [4, 4, 32, 64], 2, "h_1")
h_2 = c_layer(h_1, [3, 3, 64, 64], 1, "h_2")
with tf.variable_scope("h2_flat"):
    h_2_flat = tf.reshape(h_2, [-1, 7 * 7 * 64])

# Fully connected layers
h_3 = f_layer(h_2_flat, [7 * 7 * 64, 512], [512], True, "h_3")
q_hats = f_layer(h_3, [512, 6], [6], False, "output")

# Training
with tf.variable_scope("train"):
    q = tf.add(r, tf.mul(gamma, tf.reduce_max(q_prime)))
    q_hat = tf.gather(q_hats, a)
    mse = tf.reduce_mean(tf.clip_by_value(tf.squared_difference(q_hat, q), 0, 1))
    train_step = tf.train.AdamOptimizer(0.05).minimize(mse)

# Initialization
with tf.variable_scope("init"):
    init = tf.initialize_all_variables()


###############################################################################
# Deep Q Learning Model
###############################################################################

class DQN:

    def __init__(self, batchsize=32, warm_start=None, ckpt_dir="."):
        self.batchsize = batchsize
        self.session = tf.Session()
        self.target_session = tf.Session()
        self.saver = tf.train.Saver()
        self.ckpt_dir = ckpt_dir
        if warm_start:
            self.saver.restore(self.session,
                               os.path.join(self.ckpt_dir, warm_start))
        else:
            self.session.run(init)
        self.refresh_target_network()

    def fit(self, memory, n_updates=5):
        errs = []
        for _ in range(n_updates):
            s_, a_, r_, q_prime_ = self.gen_minibatch(memory)
            _, err = self.session.run(
                [train_step, mse],
                feed_dict = {s: s_, a: a_, r: r_, q_prime: q_prime_, gamma: .99}
            )
            errs.append(err)
        print(np.mean(errs), flush=True)

    def predict(self, data):
        return self.session.run(q_hats, feed_dict = {s: data})

    def checkpoint(self, ckpt_name):
        path = os.path.join(self.ckpt_dir, ckpt_name) 
        self.saver.save(self.session, path)

    def refresh_target_network(self, ckpt_name="refresh_target.ckpt"):
        path = os.path.join(self.ckpt_dir, ckpt_name) 
        self.checkpoint(ckpt_name)
        self.saver.restore(self.target_session, path)

    def gen_minibatch(self, memory):
        samples = memory.sample(self.batchsize)
        s_ = []
        q_prime_ = []
        a_ = []
        r_ = []
        for sample in samples:
            s_.append(np.swapaxes(sample["s"], 0, 2))
            a_.append(sample["action"])
            reward = np.zeros(6)
            reward[sample["action"]] = sample["reward"]
            r_.append(reward)
            s_prime = [np.swapaxes(sample["s_prime"], 0, 2)]
            q_hats_ = self.target_session.run(q_hats,  feed_dict = {s: s_prime})
            q_prime_.append(q_hats_)
        return s_, a_, np.squeeze(r_), np.squeeze(q_prime_)

    def close_sessions():
        self.session.close()
        self.target_session.close()
