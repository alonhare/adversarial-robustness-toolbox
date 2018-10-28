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



batch_size = 256

local_path = "C:\\Users\\alonh\\Documents\\Thesis\\adversarial-robustness-toolbox\\My implementation\\cnn_data\\"

conv1_output_reshaped = np.load(local_path + "conv1_output.npy")
conv2_output_reshaped = np.load(local_path + "conv2_output.npy")

print(conv1_output_reshaped.shape)
print(conv2_output_reshaped.shape)

max_size = conv2_output_reshaped.shape[1]
# conv1_output_reshaped = Keras.preprocessing.sequence.pad_sequences(conv1_output_reshaped,
#                                                         value=0,
#                                                         padding='post',
#                                                         maxlen=max_size)
to_add = max_size - conv1_output_reshaped.shape[1]
print("100 before: " + str(conv2_output_reshaped[100]) + " " + str(conv1_output_reshaped[100]))
print("21633 before: " + str(conv2_output_reshaped[0][21633]))
conv1_output_reshaped = np.pad(conv1_output_reshaped, [(0, 0), (0, to_add)], mode='constant')
# print(conv1_output_reshaped[1000])
# print(conv1_output_reshaped.shape)
print("100 after: " + str(conv2_output_reshaped[100]) + " " + str(conv1_output_reshaped[100]))
print("21633 after: " + str(conv2_output_reshaped[0][21633]) + " " + str(conv1_output_reshaped[0][21633]))

input_sequence = np.stack((conv1_output_reshaped, conv2_output_reshaped), axis=-1)
input_sequence = input_sequence.reshape(input_sequence.shape[0], input_sequence.shape[2], input_sequence.shape[1])
print(input_sequence.shape)
print("100 after unite: " + str(input_sequence[100][0]) + " " + str(input_sequence[100][1]))

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
model.add(LSTM(64, input_shape=(input_sequence.shape[1], input_sequence.shape[2])))
model.add(Dense(1))

# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
#
# print('Train...')
# model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=7,
#           validation_data=(x_test, y_test))
# score, acc = model.evaluate(x_test, y_test,
#                             batch_size=batch_size)
# print('Test score:', score)
# print('Test accuracy:', acc)