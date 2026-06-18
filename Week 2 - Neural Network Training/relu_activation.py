import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LeakyReLU
from tensorflow.python.keras.activations import linear, relu, sigmoid

from matplotlib.widgets import Slider
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

from lab_coffee_utils import load_coffee_data

X, Y = load_coffee_data()

print("–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––")

print(X.shape, Y.shape)

# X -> (200, 2)
# Y -> (200, 1)

# 200 examples | 2 features
# 200 binary labels | 1 column

# normalize data

print("Pre-normalization:")
print(f"Temperature; max, min: {np.max(X[:, 0])},{np.min(X[:, 0])}")
print(f"Duration; max, min: {np.max(X[:, 1])}, {np.min(X[:, 1])}")

norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X)
X_norm = norm_l(X)

print("Post-normalization")
print(f"Temperature; max, min: {np.max(X_norm[:, 0])},{np.min(X_norm[:, 0])}")
print(f"Duration; max, min: {np.max(X_norm[:, 1])}, {np.min(X_norm[:, 1])}")

# NOTE:
# A Dense layer processes one 1D example at a time from a 2D dataset.
#
# Each neuron receives the entire 1D input activation vector and computes
# a scalar output:
#
#     z_j = w_j · a_in + b_j
#
# The scalar outputs from all neurons are collected into a 1D activation
# vector for the layer.
#
# Therefore, for a Dense layer, the output activation for a single example
# remains 1D (same number of axes as the input), although its shape changes
# according to the number of neurons in the layer.

# So, input dimension/number of axes set        layer       neuron      output activ. vector
#                   2D                           1D        scalar/0D              1D


def ReLU(z):
    return max(0, z)

def Dense(a_in, W, b):

    neurons = W.shape[1]

    a_out = np.zeros(neurons)

    for j in range(neurons):

        z = np.dot(a_in, W[:, j]) + b[j]
        # is a scalar/0D

        a = ReLU(z)

        a_out[j] = a
    
    return a_out