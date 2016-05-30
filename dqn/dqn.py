from copy import deepcopy
import os

import argparse
import numpy as np
import sys
import pickle
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.regularizers import l2, l1
from keras.models import model_from_json

import theano
import theano.tensor as T


def load_model():
    model = model_from_json(open("models/cnn.json").read())
    model.load_weights("models/cnn.h5")
    return(model)

def custom_loss(y_true, y_pred):
   a = y_true.nonzero()
   zeros = y_pred - y_pred
   err = T.inc_subtensor(zeros[a], y_pred[a] - y_true[a])
   clip_err = T.clip(err, -1.0, 1.0)
   return T.sum(T.square(clip_err), axis=-1)


def calculate_y(s, a, r, s_prime, model, gamma=.99):
    r_next = model.predict(np.array([s_prime]))[0]
    r_tot = r + gamma * np.max(r_next) + .000001
    y = [r_tot if i == a else 0.0 for i in range(len(r_next))]
    return y


def init_model():

    model = Sequential()
    model.add(Convolution2D(
        input_shape = (4, 84, 84),
        nb_filter = 32,
        nb_row = 8,
        nb_col = 8,
        subsample = (4,4),
        dim_ordering='tf',
        init="glorot_uniform",
        border_mode="same",
        activation="relu"))
    #model.add(BatchNormalization())
    model.add(Convolution2D(
        #input_shape = (64, 14, 14),
        nb_filter = 64,
        nb_row = 4,
        nb_col = 4,
        subsample = (2,2),
        dim_ordering='tf',
        init="glorot_uniform",
        border_mode="same",
        activation="relu"))
    #model.add(BatchNormalization())
    model.add(Convolution2D(
        #input_shape = (64, 6, 6),
        nb_filter = 64,
        nb_row = 3,
        nb_col = 3,
        subsample = (1,1),
        dim_ordering='tf',
        init="glorot_uniform",
        border_mode="same",
        activation="relu"))
    #model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(
        #input_dim = 64 * 4 * 4,
        output_dim=512,
        init="glorot_uniform",
        activation="relu"))
    #model.add(BatchNormalization())
    model.add(Dense(
        #input_dim = 512,
        output_dim=6,
        init="glorot_uniform",
        activation="linear"))
    return model

def gen_minibatch(D, batchsize, model):
    y = []
    X = []
    for i in np.random.randint(0, len(D), batchsize*20):
        d = D[i]
        y_ = calculate_y(d["s"], int(d["action"]), int(d["reward"]),
                         np.array(d["s_prime"]), model, gamma=.99)
        X.append(d["s"])
        y.append(y_)
    return np.array(X), np.array(y)


class DQN:

    def __init__(self, batchsize=50, reset=False):
        self.batchsize = batchsize

        optimizer = RMSprop(lr=.00025)
        if reset:
            print("reseting the models")
            self.model = init_model()
            self.model.compile(loss=custom_loss, optimizer=optimizer)
        else:
            print("updating model")
            self.model = load_model()
            self.model.compile(loss=custom_loss, optimizer=optimizer)
        self.target_model = deepcopy(self.model)

    def fit(self, D, update_target_model=False):
        try:
            if update_target_model:
                self.target_model = deepcopy(self.model)
            x, y = gen_minibatch(D, self.batchsize, self.target_model)
            self.model.fit(x = x, y = y, batch_size=self.batchsize, nb_epoch=1)
        except Exception as e:
            print(e)


    def save(self):
        self.model.save_weights('models/cnn.h5', overwrite=True)
        with open("models/cnn.json", "w") as f:
           f.write(self.model.to_json())

