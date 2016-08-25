import tensorflow as tf

def weight_var(shape, name="weights"):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=name)


def bias_var(shape, name="bias"):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=name)


def c_layer(X, weight_shape, bias_shape, strides, scope):
    """Convolutional layer"""
    with tf.variable_scope(scope):
        w = weight_var(weight_shape)
        b = bias_var(bias_shape)
        c = tf.nn.conv2d(X, w, strides=strides, padding='SAME')
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


def init_model():
    with tf.Graph().as_default() as g:
        # Action, state and reward
        with tf.variable_scope("input"):
            a = tf.placeholder(tf.int32, name="a")
            s = tf.placeholder(tf.float32, shape=[None, 84, 84, 4], name="s")
            r = tf.placeholder(tf.float32, shape=[None, 6], name="r")
            q_prime = tf.placeholder(tf.float32, shape=[None, 6], name="q_prime")
            gamma = tf.placeholder(tf.float32, name="gamma")


        # Convolutional layers
        h_0 = c_layer(s, [8, 8, 4, 32], [32], [1, 4, 4, 1], "h_0")
        h_1 = c_layer(h_0, [20, 20, 32, 64], [64], [1, 2, 2, 1], "h_1")
        h_2 = c_layer(h_1, [9, 9, 64, 64], [64], [1, 1, 1, 1], "h_2")

        # Fully connected layers
        h_2_flat = tf.reshape(h_2, [-1, 11 * 11 * 64])
        h_3 = f_layer(h_2_flat, [11 * 11 * 64, 512], [512], True, "h_3")
        q_hats = f_layer(h_3, [512, 6], [6], False, "output")

        # Loss
        with tf.variable_scope("init"):
            init = tf.initialize_all_variables()
        with tf.variable_scope("loss"):
            q = tf.add(r, tf.mul(gamma, tf.reduce_max(q_prime)))
            q_hat = tf.gather(q_hats, a)
            mse = tf.reduce_mean(tf.squared_difference(q_hat, q))
            train_step = tf.train.AdamOptimizer(0.05).minimize(mse)

        return g

if __name__ == "__main__":
    for n in init_model().as_graph_def().node:
        print(n)
    #print(init_model())
