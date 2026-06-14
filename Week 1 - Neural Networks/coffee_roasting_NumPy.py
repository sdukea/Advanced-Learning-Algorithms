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
    # a neural network layer can have as input data/activations of MANY/ANY DIMENSION
    # and everything - parameters W and b and the output activation (a_out) - will vary
    # accordingly.

    # Here, the input is a single example, not 2D data/table/matrix
    # and so, if we have input of shape (n,), then DIMENSION = 1/a 1D array

    # then, if we have, say 3 neurons, and

    # we have input shape = (n,) - then

    # 1. W shape = (n,3)
    # 2. b shape = (3,)
    # 3. a_out/output activation shape = (3,)

    # because, if input data was 2D/2-Dimensional:
    # (consider 3 neurons itself)
    # (consider 1 feature here)

    # input:
    # [[200.0],
    #  [150.0]]
    # a 2D input data of 2 examples

    # then a_out would be:

    # [[a1_(1), a2_(1), a3_(1)], <-- example 1/200.0
    #  [a1_(1), a2_(1), a3_(1)]] <-- example 2/150.0
    # HAS THE SAME DIMENSION - 2D

    # and even if you had input:
    # [[200.0]]
    # which is still a 2D input data of 1 example

    # a_out would be:

    # [[a1_(1), a2_(1), a3_(1)]]

    # but, if input was 1-Dimensional:
    # [200.0]
    # now, data is just 1 example - a 1D array and not a table

    # and so, a_out would be:

    # [a1_(1), a2_(1), a3_(1)]

    # which has shape of just (3,) -> (number of neurons,)

    # exactly being 1-Dimensional as the input data

    # So, the DIMENSION of input and a_out/activation vector of a layer would have the same 
    # dimension always/same number of axes.
    # If input has 2D shape, then the output activ. vector/a_out will also have a 2D shape

    # On the other hand,

    # W dimension can vary

    # If the input has n features and the layer has j neurons:

    # W.shape = (n, j)

    # The n rows correspond to input features.
    # The j columns correspond to neurons.

    # Therefore, even if the input activation is a 1D vector of shape (n,),
    # W must still be a 2D matrix because each neuron needs a weight
    # for every input feature.

    # and for b, the shape depends ONLY on the number of neurons in the layer

    # NOTE:

    # As you know,
    # A Dense layer is defined on a single example - that's why a layer processes
    # one example at a time

    # Case 1: when input data/activation vector has a 2D shape - a table/2D matrix

    # So a Dense layer here will look at one example/row at a time from the 2D matrix
    # i.e. it will process the 1D array/example from the 2D matrix at a time
    # Because in a 2D matrix/table as data, each row/example is a 1D array
    
    # So, inherently, if you have parameters for each neuron like (say layer 1):
    # w1_(1), w2_(1), w3_(1) and so on, and
    # b1_(1), b2_(1), b3_(1)

    # each neuron does this:
    # z1_(1) = w1_(1) * 1D-example + b1_(1)
    # z2_(1) = w2_(1) * 1D-dexample + b2_(1)
    # z3_(1) = w3_(1) * 1D-example + b3_(1)

    # this would now be correspond to one example/row in the output activation vector of this neuron
    # which is a_(1):
    # a_(1) = [[z1_(1), z2_(1), z3_(1)]] <-- for first example
    #                   .
    #                   .
    #                   .
    # and so on if you have more 1D examples

    # NOTE: both input data/activation vector to this layer and the output activ. vector have
    # the same dimension - both are 2D

    # Here:
    # input data -> 2D table/matrix
    # each example -> should be 1D array

    # The natural case:
    # Now, naturally, a layer will see one example at a time
    # When input data is of 3D shape, like giving in images -> (1000, 28, 28) -> 1000 images of 
    # 28 x 28 px size, then each example/image has a 2D dimension
    # And when it comes to input data of a 2D shape, then each example has a 1D array

    # So, if you specified the dimension of the input data that is coming into the layer

    units = W.shape[1]
    # we infer the number of neurons from shape of W

    a_out = np.zeros(units)

    for j in range(units): # for each neuron

        w_neuron = W[:, j] 

        z_neuron = np.dot(w_neuron, a_in) + b[j]

        a_out[j] = g(z_neuron)
    
    return (a_out)

def Sequential(x, W1, b1, W2, b2):
    
    # W1, b1 -> parameters for layer 1
    # W2, b2 -> parameters for layer 2

    a1 = Dense(x, W1, b1)
    a2 = Dense(a1, W2, b2)

    # initialize the 2 layers using the parameters
  
    return a2

# layer 1 parameters
W1_tmp = np.array( [[-8.93,  0.29, 12.9], [-0.1,  -7.32, 10.81]])
b1_tmp = np.array( [-9.82, -9.28,  0.96])

# so this means:
# 1. we have two features
# 2. we have three neurons

# layer 2 parameters
W2_tmp = np.array( [[-31.18], [-27.59], [-32.56]] )
b2_tmp = np.array( [15.41] )

# means:
# 1. we have 3 features
# 2. we have 1 neuron

def predict(X, W1, b1, W2, b2):
    m = X.shape[0]

    predictions = np.zeros(W2.shape[0],W2.shape[1])

    for i in range(m):
        pass
    
    