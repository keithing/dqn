from copy import copy
import os

import argparse
import numpy as np
import pickle
from keras.optimizers import RMSprop
from keras.models import model_from_json

from dqn.cnn import init_model, custom_loss


def approximate_q(action, reward, s_prime, model, gamma=.99, num_actions=6):
    if s_prime[-1] is None:
        r_next = [0]
    else:
        r_next = model.predict(np.array([s_prime]))[0]
    y = [0.0] * num_actions
    y[action] = reward + gamma * np.max(r_next)
    return y


def gen_minibatch(memory, batchsize, target_model):
    samples = memory.sample(batchsize)
    y = []
    X = []
    for sample in samples:
        q = approximate_q(sample["action"], sample["reward"],
                          sample["s_prime"], target_model, .99)
        X.append(sample["s"])
        y.append(q)
    return X, y


class DQN:

    def __init__(self, batchsize=32, model_json=None, model_h5=None):
        self.batchsize = batchsize
        self.model_json = model_json or os.path.join("models", "cnn.json")
        self.model_h5 = model_h5 or os.path.join("models", "cnn.h5")
        self.optimizer = RMSprop(lr=.000001)
        if os.path.exists(self.model_json) and os.path.exists(self.model_h5):
            print("Loading existing model.")
            self.model = self.load_model()
        else:
            print("Initializing a new model.")
            self.model = init_model()
        self.model.compile(loss=custom_loss, optimizer=self.optimizer)
        self.target_model = copy(self.model)

    def fit(self, memory, update_target_model=False, n_updates=5):
        if update_target_model:
            self.target_model = copy(self.model)
        X, Y = gen_minibatch(memory, self.batchsize*n_updates, self.target_model)
        self.model.fit(X, Y, batch_size=self.batchsize, verbose=2, nb_epoch=1)

    def load_model(self):
        model = model_from_json(open(self.model_json).read())
        model.load_weights(self.model_h5)
        return(model)

    def save(self, model_json=None, model_h5=None):
        json_out = model_json or self.model_json
        h5_out = model_h5 or self.model_h5
        self.model.save_weights(h5_out, overwrite=True)
        with open(json_out, "w") as f:
           f.write(self.model.to_json())

