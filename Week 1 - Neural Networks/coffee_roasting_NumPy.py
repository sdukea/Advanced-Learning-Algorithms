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
    # and everything - parameters W and b and the output activation (a_out) - can vary.

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
    
    # NOTE: actual input data - data that contains many examples as input to the layer/model
    # and not just 1 example

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
    # and so on if you have more 1D examples in the input data of 2D matrix

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
    # i.e. if you tell that input data is 1 example and has shape (n,) - 1 dimensional,

    # THEN,

    # We can set the actual data as input to be a 2D shape ONLY
    # Essentially, we are demanding/setting that the actual input data is a 2-dimensional matrix/
    # table by specifying that the dimension of a single example is 1-dimensional, which is
    # what the layer will process at a time from the actual input data

    # Because each layer sees 1 example at a time from the actual data
    # And if we specify that the layer will see/process a 1-dimensional array of shape (n,) from the
    # actual input data, then this input data is a container of these 1-D examples only, which is
    # inherently a 2D matrix/table (a container of 1D arrays/examples)

    # So in order to demonstrate how a layer will work, we do this:
    # we make the Dense layer see only one example
    # and in our case, this one example is 1D - so, the actual input dataset should be 2D

    units = W.shape[1]
    # we infer the number of neurons from shape of W

    a_out = np.zeros(units)
    # for one example -> one row in the activ. vector

    for j in range(units): # for each neuron

        w_neuron = W[:, j]

        z_neuron = np.dot(w_neuron, a_in) + b[j]

        a_out[j] = g(z_neuron)
    
    # The process here is to show you:
    # how a layer will process example by example
    # and so we have one example passed in

    # You normally give in a batch/actual input data that has examples in it
    # and TF will dynamically see through the examples and process them one by one - in a layer
    # Here, you will have to write handwritten code to take in one example into a layer
    # and produce the output activation vector (based on the number of units and this one example -
    # as shape of output activ. vector = (number of examples, number of units))
    # |
    # NOTE: Here, input data is one example of 1D, shape = (n,)
    # So, actual input data should be 2D, shape = (m,n)

    # So, output activ. vector would be be 1D, shape = (number of units,)
    # actual output activ. vector would be 2D, shape = (number of examples, number of units)

    return (a_out)

def Sequential(x, W1, b1, W2, b2):
    
    # W1, b1 -> parameters for layer 1
    # W2, b2 -> parameters for layer 2
    # so we have a model/network of two layers

    a1 = Dense(x, W1, b1)
    # single example input

    a2 = Dense(a1, W2, b2)

    # initialize the 2 layers using the parameters
  
    return a2

# All 'Sequential' does: string multiple layers by getting arguments for:
# 1. W for each layer
# 2. b for each layer

# layer 1 parameters
W1_tmp = np.array( [[-8.93,  0.29, 12.9],   # <-- w1_(1)
                    [-0.1,  -7.32, 10.81]]) # <-- w2_(1)
b1_tmp = np.array( [-9.82, -9.28,  0.96])   # <-- b1_(1)

# so this means:
# 1. we have 2 features -> so the 1 single example as input has shape (2,)/(number of features,)
# 2. we have 3 neurons

# layer 2 parameters
W2_tmp = np.array( [[-31.18],   # <-- w1_(2)
                    [-27.59],   # <-- w2_(2)
                    [-32.56]] ) # <-- w3_(2)
b2_tmp = np.array( [15.41] )    # <-- b1_(2)

# means:
# 1. we have 3 features -> layer 2 recieves from layer 1 - a1/a_(1) - and will have shape (3,)/
# (number of units,)
# 2. we have 1 neuron

# So NOTE:
# as the final layer - layer 2 - has one neuron,
# the output of the network is just a 2D vector (as we will be getting 2D input data when
# predicting and the output activation vector (final) should also be a 2D dimension) of 1 value
# So, shape=(1,1) when data is 2D like (m,n) - so, 1 example, 1 unit
# This MATCHES!

# NOTE:
# When we did: for j in range(units) - line 192
# This is what we mean when we say the layer i.e. each neuron will look at one example from data
# So, for each unit/neuron in layer (say layer 1):
#   1. get the weight for this neuron - from W
#   2. get the bias for this neuron - from b
#   3. execute model/matrix multiplication between
#      weight for neuron neuron * example + b for this neuron

# And then we aggregate each neuron's output for 1 example into a_out -> becomes 1 row

# And we repeat (line 256 to 262) for each example in the input data for this layer: layer 1

def predict(X, W1, b1, W2, b2):

    m = X.shape[0]
    # Now, we get actual input data - data with many examples in it
    # and not data that is just 1 example

    predictions = np.zeros((m,1))
    # Input incoming data is 2D
    # So initialize the final predictions array to also be 2D.

    for i in range(m):
        predictions[i, 0] = Sequential(X[i], W1, b1, W2, b2)
        #                   Feed in example by example to the network/model/Sequential
        #                   as this is what we're handwritten the code for

    return predictions