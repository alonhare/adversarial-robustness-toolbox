# Dependencies for entire notebook here
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


import keras.backend as k
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout
import numpy as np
from keras.models import model_from_yaml

from art.attacks.fast_gradient import FastGradientMethod
from art.classifiers import KerasClassifier
from art.utils import load_dataset
from keras.models import Model

# Read MNIST dataset
(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset(str('mnist'))

local_path = "C:\\Users\\alonh\\Documents\\Thesis\\adversarial-robustness-toolbox\\My implementation\\cnn_data\\"
# should start run from here
yaml_file = open(local_path + "model.yaml", 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights(local_path + "model.h5")

loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# sanity check
# classifier = KerasClassifier((min_, max_), model=loaded_model)
# preds = np.argmax(classifier.predict(x_test), axis=1)
# acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
# print("\nTest accuracy: %.2f%%" % (acc * 100))

input_shape = x_train[0].shape

model = loaded_model

def getModelOutput(model, layer_name, data):
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(data)
    return intermediate_output


conv1_output = getModelOutput(model, "conv1", x_test)
conv2_output = getModelOutput(model,"conv2", x_test)

print("conv1: ")
print(conv1_output)
print("conv2: ")
print(conv2_output)
print("conv1 output shape: ")
print(conv1_output.shape)
print("conv2 output shape: ")
print(conv2_output.shape)