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
      a_out (ndarray (j,))  : j units
    """             

    # NOTE:
    # Generally, the input to a neural network would be 2D/matrix/table
    # where we have (m,n) -> the number of examples m and number of features n
    # from this:
    # 1. the number of feature are inferred to find shape/values of parameter W for the layer
    #    as shape=(number of features, number of units)
    # 2. the number of examples can be traced to the shape of the output activation vector as it
    #    has shape=(number of examples, number of units)

    # But you have to know that:
    # a neural network layer can have data/activations of ANY SHAPE
    # and everything - parameters W and b and the output activation (a_out) - will vary
    # accordingly.

    # Here, the input is a single example, not 2D data/table/matrix
    # and so, if we have input of shape = (n,)

    # then, if we have, say 3 neurons, and

    # we have input shape = (n,) - then

    # 1. W shape = (n,3)
    # 2. b shape = (3,)
    # 3. a_out/output activation shape = (3,)

    # because, if input was 2D data like:
    # (consider 3 neurons itself)
    # (consider 1 feature here)

    # input:
    # [[200.0],
    #  [150.0]]
    # a 2D input data of 2 examples

    # then a_out would be:

    # [[a1_(1), a2_(1), a3_(1)], <-- example 1/200.0
    #  [a1_(1), a2_(1), a3_(1)]] <-- example 2/150.0

    # and even if you had input:
    # [[200.0]]
    # which is still a 2D input data of 1 example

    # a_out would be:

    # [[a1_(1), a2_(1), a3_(1)]]

    # but, if input was:
    # [200.0]
    # now, data is just 1 example - a 1D array and not a table

    # and so, a_out would be:

    # [a1_(1), a2_(1), a3_(1)]

    # which has shape of just (3,) -> (number of neurons,)

    # exactly being a 1D array like the input

    # So, the shape of input and the a_out depend on each other 
    # i.e. they will have the same dimension

    # Shape of W will vary as the shape also specifies/hints at the number of neurons
    # and so it is not neccessarily the case where the shape of input matches the shape of
    # W

    # And the shape of b depends on the number of neurons solely - so need not consider for
    # discussion

    units = W.shape[1]
    # get number of neurons

    a_out = np.zeros(units)

    for j in range(units): # for each neuron

        w_neuron = W[:, j] 

        z_neuron = np.dot(w_neuron, a_in) + b[j]

        a_out[j] = g(z_neuron)
    
    return (a_out)