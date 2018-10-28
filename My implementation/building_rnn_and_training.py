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
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout, LSTM
import numpy as np
from keras.models import model_from_yaml

from art.attacks.fast_gradient import FastGradientMethod
from art.classifiers import KerasClassifier
from art.utils import load_dataset
from keras.models import Model

batch_size = 128

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

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

model = Sequential()
model.add(LSTM(128, input_shape=(input_sequence.shape[1], input_sequence.shape[2])))
# model.add(Dropout(0.25))
# model.add(Dense(1024, activation='relu'))
# model.add(Dropout(0.25))
model.add(Dense(1, activation='softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=7,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score*100)
print('Test accuracy:', acc*100)