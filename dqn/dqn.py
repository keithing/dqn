import os

import argparse
import numpy as np
import sys
import pickle
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.regularizers import l2, l1
from keras.models import model_from_json


def load_model():
    model = model_from_json(open("models/cnn.json").read())
    model.load_weights("models/cnn.h5")
    return(model)


def calculate_y(s, a, r, s_prime, model, gamma=.99):
    y = model.predict(np.array([s]))[0]
    y[a] = r + gamma * np.max(model.predict(np.array([s_prime])))
    y = np.round(y, 6)
    return y


def init_model():

    # Params
    lay1 = Convolution2D(
        input_shape = (3, 105, 80),
        nb_filter = 32,
        nb_row = 9,
        nb_col = 8,
        subsample = (4,4),
        dim_ordering='tf',
        init="glorot_uniform",
        border_mode="same",
        W_regularizer=l2(0.01),
        activation="relu")

    lay2 = Convolution2D(
        #input_shape = (32, 25, 18),
        nb_filter = 64,
        nb_row = 5,
        nb_col = 4,
        subsample = (2,2),
        dim_ordering='tf',
        init="glorot_uniform",
        border_mode="same",
        W_regularizer=l2(0.01),
        activation="relu")

    lay3 = Convolution2D(
        #input_shape = (64, 11, 8),
        nb_filter = 64,
        nb_row = 3,
        nb_col = 3,
        subsample = (1,1),
        dim_ordering='tf',
        init="glorot_uniform",
        border_mode="same",
        W_regularizer=l2(0.01),
        activation="relu")

    lay4 = Flatten()

    lay5 = Dense(
        #input_dim = 64 * 9 * 6 * 2,
        output_dim=512,
        init="glorot_uniform",
        W_regularizer=l1(0.01),
        activation="relu")

    lay6 = Dense(
        #input_dim = 512,
        output_dim=6,
        init="glorot_uniform",
        W_regularizer=l1(0.01),
        activation="linear")


    model = Sequential()
    model.add(lay1)
    model.add(lay2)
    model.add(lay3)
    model.add(lay4)
    model.add(lay5)
    model.add(lay6)
    return model


class DQN:

    def __init__(self, batchsize=50, n_samples=400, reset=False):
        self.batchsize = batchsize
        self.n_samples = n_samples

        optimizer = RMSprop(lr=.00001)
        if reset:
            print("reseting the models")
            self.model = init_model()
            self.model.compile(loss="mean_squared_error", optimizer=optimizer)
        else:
            print("updating model")
            self.model = load_model()
            self.model.compile(loss="mean_squared_error", optimizer=optimizer)

    def fit(self, D):
        try:
            self.model.fit_generator(
                self.generator(D),
                samples_per_epoch = self.n_samples,
                nb_epoch = 1,
                max_q_size = 10)
        except Exception as e:
            print(e)


    def generator(self, D):
        while True:
            y = []
            s = []
            for i in range(self.batchsize):
                d = D[i % len(D)]
                y_ = calculate_y(d["s"],
                                 int(d["action"]),
                                 int(d["reward"]),
                                 np.array(d["s_prime"]),
                                 self.model,
                                 gamma=.9)
                s.append(d["s"])
                y.append(y_)
            yield np.array(s), np.array(y)


    def save(self):
        self.model.save_weights('models/cnn.h5', overwrite=True)
        with open("models/cnn.json", "w") as f:
           f.write(self.model.to_json())

