import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from lab_utils_common import dlc, sigmoid
from lab_coffee_utils import load_coffee_data

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

# use same dataset

X, Y = load_coffee_data()

print(X.shape, Y.shape)

# normalize data

print(f"Temperature Max, Min pre normalization: {np.max(X[:,0]):0.2f}, {np.min(X[:,0]):0.2f}")
print(f"Duration    Max, Min pre normalization: {np.max(X[:,1]):0.2f}, {np.min(X[:,1]):0.2f}")

norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X)

X_norm = norm_l(X)

# model from scratch (forward prop)

g = sigmoid

def Dense(a_in, W, b):
    """
    Computes dense layer
    Args:
      a_in (ndarray (n, )) : Data, 1 example 
      W    (ndarray (n,j)) : Weight matrix; (number of features in example, number of neurons/units)
      b    (ndarray (j, )) : bias vector, (number of neurons/units,)
    Returns
      a_out (ndarray (j,))  : j units|
    """

    # NOTE:
    # here's what you should understand:
    # model = tf.keras.Sequential([
    # tf.keras.Input(shape=(2,)),
    # tf.keras.layers.Dense(3, activation='sigmoid', name='L1'),
    # tf.keras.layers.Dense(1, activation='sigmoid', name='L2')
    # ])

    # take this model initialization for example
    # (from coffee_roasting_TF)

    # 1. the Input layer
    # it expects 2 features in the example it sees from data i.e. a 1D array with two values
    # that has shape = (2,)
    # and so input data/test data can be of shape (None, 2)

    # say we pass in this data:
    #X = np.array([
    # [200, 13], <--- example 1
    # [220, 15], <--- example 2
    # [180, 11]  <--- example 3
    # ])
    
    # its shape is (3,2) - each eg. has 2 features and each eg. is a 1D array -> MATCHES!

    # now,

    # 2. L1
    # this data enters L1 where this layer has
    # 3 neurons
    # sigmoid activation
    # parameters are initialized here
    # so, shape of W = (2, 3) -> (number of features, number of units)
    # and it may look something like:

    # W = np.array([w1_1_val_1, w1_2_val_1], <--- feature 1
    #              [w1_1_val_2, w1_2_val_2], <--- feature 2
    #              [w1_1_val_3, w1_2_val_3]) <--- feature 3
    #                   ^           ^
    #                Neuron 1    Neuron 2
    
                    
    
    units = W.shape[1]
    # get number of neurons

    a_out = np.zeros(units)

    for j in range(units): # for each neuron

        w_neuron = W[:, j]

        z_neuron = np.dot(w_neuron, a_in) + b[j]

        a_out[j] = g(z_neuron)
    
    return (a_out)

