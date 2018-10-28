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

sess = tf.InteractiveSession()


local_path = "C:\\Users\\alonh\\Documents\\Thesis\\adversarial-robustness-toolbox\\My implementation\\cnn_data\\"
# should start run from here
yaml_file = open(local_path + "model.yaml", 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights(local_path + "model.h5")


conv1 = loaded_model.layers[0].get_weights()
conv2 = loaded_model.layers[1].get_weights()

W_conv1 = conv1[0]
b_conv1 = conv1[1]

W_conv2 = conv2[0]
b_conv2 = conv2[1]

print("woho")

sess.close()