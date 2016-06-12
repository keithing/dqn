from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
import theano
import theano.tensor as T

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


def custom_loss(y_true, y_pred):
   err = y_pred[y_true.nonzero()] - y_true.nonzero_values()
   clip_err = T.clip(err, -1.0, 1.0)
   return T.mean(T.square(clip_err))
