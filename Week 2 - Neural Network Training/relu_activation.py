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

def Dense_(a_in, W, b):

    neurons = W.shape[1]

    a_out = np.zeros(neurons)

    for j in range(neurons):

        z = np.dot(a_in, W[:, j]) + b[j]
        # is a scalar/0D

        a = ReLU(z)

        a_out[j] = a
    
    return a_out

# initialize a network with 2 layers 
def Sequential_(x, W1, b1, W2, b2):
    a1 = Dense_(x, W1, b1)
    a2 = Dense_(a1, W2, b2)

    return a2

# NOTE:
# We are building a NETWORK that works by taking in a single 1D example as input
# So that when we 'predict' - as you see in line 286 on 'coffee_roasting_NumPy.py' - we
# do predictions for each example from the 2D input (actual) as each example in a 2D input is a 1D
# example, for which we've already built a NETWORK that works on processing 1D example inputs and 
# iterating through the 2D matrix input (actual), we serve the the 1D examples from these as
# the input to the NETWORK we have built.
# |
# Why are we doing this:
# Because we are trying to imitate what TF does actually under the hood.
# model = tf.keras.Sequential([
#     tf.keras.Input(shape=(2,)), <--------------------------------------|
#     tf.keras.layers.Dense(3, activation='sigmoid', name='L1'),         |
#     tf.keras.layers.Dense(1, activation='sigmoid', name='L2')          |
# ])                                                                     |
# You say that the input dimension of data i.e. the shape of ONE EXAMPLE ^ is 1D/(just 2 features)

# NOTE: If I say that the dimension is 1D, then it can have any shape/features inside: (1,), (2,) 
# and so on - 1D input data makes it evident that the dimension is 1D and there are these-many 
# features in it.
# If input_dimension is (28,28) - 2D - then the input dimension is 2D and the number of features
# are 28 x 28 = 784 features.

# And TF now can take in any 2D data (as you know it - as input dimension is 1D) and it will
# dynamically iterate through each example under the hood - with the input/set parameters - and
# give out predictions.

# So, say you have input 2D data of shape to your model: (200, 2)

# This is CORRECT/APPLICABLE - data is in a 2D format and each example is also 1D and has 2 features

# And you have 2 layers in your model - layer 1 and 2

# layer 1 -> 3 neurons (the layer that takes the input)
# layer 2 -> 2 neurons
# (and maybe an output layer as well - output layer)

# NOTE: (),{} -> represents scalar values

# layer 1 W shape would be: (2, 3)
# and it would be something like:
# W = [[ ()  ()  ()], <- feature 1
#      [ ()  ()  ()]] <- feature 2
#        ^   ^   ^
# unit   1   2   3 

# So, this is how - typically - the weights are assigned. And of course:

# b = [()  ()  ()]
#      ^   ^   ^
# unit 1   2   3
# shape: (3,)

# So, what happens in TF is that:

# for each example from data - which has a 1D shape and has 2 features:
# |
# now in layer 1 that takes this 1 example from input data (actual) as input:
# |
# and for each neuron:
# example * unit 1 W + unit 1 b
# |
# next neuron:
# example * unit 2 W + unit 2 b
# |
# next neuron:
# example * unit 3 W + unit 3 b

# It's the SAME first example (out of the 200 example in actual input data)
# that passes through the ENTIRE network/model to register in a_out/final output layer activation
# vector as the first row

# NOTE: 'passes through the ENTIRE network' means that it turns richer and richer as it moves
# along the network but is still the first example that had been initiated/taken as input first
# from actual 2D input data

# as a_out/layer's output activation vector has shape: (number of examples, number of neurons)

# if the output layer had, say 1 neuron, then:

# a_out = [[()]] <- the first row = the first example from input data that passed through - turning
# richer - and ended up here

# for the next example, out of the 200 examples, it will pass through the entire network/model
# to end up below/as the second row

# a_out = [[()], <- first example being passed through/first row
#          [{}]] <- second example being passed through/second row

# and so on for every example taken - one at a time

# But, TF does all of this under the hood.

# If you intialized the model and return the predictions:

# predictions = model.predict(X_test)

# predictions would have the shape: (200, 1)

# for each and every 200 example in the layer, that 1 neuron gave predictions.

# All the dot products and output processing of every layer happens dynamically and
# under the hood.

# So, when you are building the 