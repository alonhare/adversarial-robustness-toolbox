# Dependencies for entire notebook here
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import keras as Keras
import keras.backend as k
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout, LSTM, Bidirectional
import numpy as np
from keras.models import model_from_yaml

from art.attacks.fast_gradient import FastGradientMethod
from art.classifiers import KerasClassifier
from art.utils import load_dataset
from keras.models import Model

batch_size = 64

local_path = "C:\\Users\\alonh\\Documents\\Thesis\\adversarial-robustness-toolbox\\My implementation\\cnn_data\\"

input_sequence = np.load(local_path + "input_sequence.npy")

input_size = input_sequence.shape[0]

y = np.ones(shape=(input_size,1))

train_part = 0.7
train_size = int(input_size * train_part)

x_train = input_sequence[:train_size]
y_train = y[:train_size]
x_test = input_sequence[train_size: input_size]
y_test = y[train_size: input_size]

np.set_printoptions(threshold=np.inf)
print(x_train[0])
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)