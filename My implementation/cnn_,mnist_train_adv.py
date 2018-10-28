# -*- coding: utf-8 -*-
"""Trains a convolutional neural network on the MNIST dataset, then attacks it with the FGSM attack."""
from __future__ import absolute_import, division, print_function, unicode_literals

from os.path import abspath
import sys
sys.path.append(abspath('.'))

import keras.backend as k
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout
import numpy as np

from art.attacks.fast_gradient import FastGradientMethod
from art.attacks.deepfool import DeepFool
from art.classifiers import KerasClassifier
from art.utils import load_dataset

# Read MNIST dataset
(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset(str('mnist'))
print(x_train.shape)
# Create Keras convolutional neural network - basic architecture from Keras examples
# Source here: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
k.set_learning_phase(1)

model = Sequential()
conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:], name="conv1")

model.add(conv1)
conv2 = Conv2D(64, (3, 3), activation='relu', name="conv2")

model.add(conv2)
maxPool = MaxPooling2D(pool_size=(2, 2))
model.add(maxPool)
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

classifier = KerasClassifier((min_, max_), model=model)
classifier.fit(x_train, y_train, nb_epochs=5, batch_size=128)

# Evaluate the classifier on the test set
preds = np.argmax(classifier.predict(x_test), axis=1)
acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
print("\nTest accuracy: %.2f%%" % (acc * 100))

# untill here cnn

#



# Craft adversarial samples with FGSM
epsilon = .1  # Maximum perturbation
adv_crafter = FastGradientMethod(classifier)
x_test_adv = adv_crafter.generate(x=x_test, eps=epsilon)
print(x_test_adv.shape)

local_path = "C:\\Users\\alonh\\Documents\\Thesis\\adversarial-robustness-toolbox\\My implementation\\cnn_data\\"
np.save(local_path + "adv_img_list_FGSM.npy", x_test_adv)
print("first")
# reset and restore old variables
model_yaml = model.to_yaml()
with open(local_path + "model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights(local_path + "model.h5")
print("third")