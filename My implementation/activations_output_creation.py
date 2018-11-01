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
# # preds = np.argmax(classifier.predict(x_test), axis=1)
# # acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
# # print("\nTest accuracy: %.2f%%" % (acc * 100))

input_shape = x_train[0].shape

model = loaded_model

def getModelOutput(model, layer_name, data):
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(data)
    return intermediate_output


conv1_output = getModelOutput(model, "conv1", x_test)
conv2_output = getModelOutput(model,"conv2", x_test)

adv_img_list_FGSM = np.load(local_path + "adv_img_list_FGSM.npy")


conv1_output_adv = getModelOutput(model, "conv1", adv_img_list_FGSM)
conv2_output_adv = getModelOutput(model,"conv2", adv_img_list_FGSM)

# print("conv1: ")
# print(conv1_output)
# print("conv2: ")
# print(conv2_output)
print("conv1 output shape: ")
print(conv1_output.shape)
print("conv2 output shape: ")
print(conv2_output.shape)
print(type(conv1_output))

conv1_output = conv1_output.reshape(conv1_output.shape[0], conv1_output.shape[1]*conv1_output.shape[2]*conv1_output.shape[3])
conv2_output = conv2_output.reshape(conv2_output.shape[0], conv2_output.shape[1]*conv2_output.shape[2]*conv2_output.shape[3])

conv1_output_adv = conv1_output_adv.reshape(conv1_output_adv.shape[0], conv1_output_adv.shape[1]*conv1_output_adv.shape[2]*conv1_output_adv.shape[3])
conv2_output_adv = conv2_output_adv.reshape(conv2_output_adv.shape[0], conv2_output_adv.shape[1]*conv2_output_adv.shape[2]*conv2_output_adv.shape[3])

# print("conv1: ")
# print(conv1_output)
# print("conv2: ")
# print(conv2_output)
print("conv1 output shape: ")
print(conv1_output.shape)
print("conv2 output shape: ")
print(conv2_output.shape)
print(type(conv1_output))


def reshape_2Con_to_one(conv1_output_reshaped, conv2_output_reshaped):
    if conv2_output_reshaped.shape[1]>conv1_output_reshaped.shape[1]:
        to_add = conv2_output_reshaped.shape[1] - conv1_output_reshaped.shape[1]
        # print("before conv1:" + str(conv1_output_reshaped[100]))
        conv1_output_reshaped = np.pad(conv1_output_reshaped, [(0, 0), (0, to_add)], mode='constant')
        # print("after conv1:" + str(conv1_output_reshaped[100]))
    else:
        to_add = conv1_output_reshaped.shape[1] - conv2_output_reshaped.shape[1]
        # print("before conv2:" + str(conv2_output_reshaped.shape))
        conv2_output_reshaped = np.pad(conv2_output_reshaped, [(0, 0), (0, to_add)], mode='constant')
        # print("after conv2:" + str(conv2_output_reshaped[100].shape))


    input_sequence = np.stack((conv1_output_reshaped, conv2_output_reshaped), axis=-1)
    # print("100 before reshape: " + str(input_sequence[100]))
    # print("before reshape: " + str(input_sequence.shape))
    input_sequence = np.transpose(input_sequence, (0, 2, 1))
    # print("after reshape: " + str(input_sequence.shape))
    # print("100 after reshape: " + str(input_sequence[100]))
    # print("100 after unite: conv1:" + str(input_sequence[100][0]) + " conv2:" + str(input_sequence[100][1]))
    return input_sequence
np.save(local_path + "input_sequence.npy", reshape_2Con_to_one(conv1_output, conv2_output))
np.save(local_path + "input_sequence_adv.npy", reshape_2Con_to_one(conv1_output_adv, conv2_output_adv))
