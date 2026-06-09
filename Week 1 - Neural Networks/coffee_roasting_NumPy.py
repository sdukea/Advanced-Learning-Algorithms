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
    
    units = W.shape[1]
    # get number of neurons

    a_out = np.zeros(units)

    for j in range(units): # for each neuron

        w_neuron = W[:, j]

        z_neuron = np.dot(w_neuron, a_in) + b[j]

        a_out[j] = g(z_neuron)
    
    return (a_out)
