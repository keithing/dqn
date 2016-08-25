from copy import copy
import os

import argparse
import numpy as np

from dqn.cnn2 import init_model

def gen_minibatch(memory, batchsize, tf_session):
    samples = memory.sample(batchsize)
    s = []
    q_prime = []
    a = []
    r = []
    for sample in samples:
        s.append(sample["s"])
        a.append(sample["action"])
        r.append(sample["reward"])
        s_prime = [np.swapaxes(sample["s_prime"], 0, 2)]
        q_prime.append(
            tf_session.run("output/add",  feed_dict = {"input/s:0": s_prime}))
    return s, a, r, q_prime


class DQN:

    def __init__(self, batchsize=32, model_json=None, model_h5=None):
        self.batchsize = batchsize
        self.model_json = model_json or os.path.join("models", "cnn.json")
        self.model_h5 = model_h5 or os.path.join("models", "cnn.h5")
        if os.path.exists(self.model_json) and os.path.exists(self.model_h5):
            print("Loading existing model.")
            self.model = self.load_model()
        else:
            print("Initializing a new model.")
            self.model = init_model()
        #self.target_model = copy(self.model)

    def fit(self, memory, tf_session, update_target_model=False, n_updates=5):
        if update_target_model:
            #self.target_model = copy(self.model)
            pass
        for _ in range(n_updates):
            batch = gen_minibatch(memory, self.batchsize, tf_session)
            s, a, r, q_prime = batch
            print(q_prime)
            err = tf_session.run("loss/Adam",
                feed_dict = {s: s, a: a, r: r, q_prime: q_prime, gamma: .99})
            print(err, flush=True)

    def load_model(self):
        pass
        #LOAD

    def save(self, model_json=None, model_h5=None):
        pass
        #SAVE
