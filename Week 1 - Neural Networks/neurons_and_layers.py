import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.python.keras.activations import sigmoid

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# with this, TF supresses INFO messages and WARNING messages
# and only displays ERROR messages

# instead of:

# INFO:tensorflow: Created device GPU:0
# WARNING:tensorflow: Some operation may run slowly

# which are just updates, you only get ERROR messages - actual errors and all the
# informational messages and warning are supressed


# neuron without activation - regression/linear model

X_train = np.array([[1.0], 
                    [2.0]], dtype=np.float32)

# you have training examples in the format of an actual table/2D matrix
# you see ROWS now, which are training examples just like in real-life data - one single feature 
# in our case

# X_train is of shape (2,1) - a table/2D matrix

y_train = np.array([[300.0], 
                    [500.0]], dtype=np.float32)

# again - you see ROWS
# and y_train is a 2D matrix of shape (2, 1)

# data in TF is represented in a different way than what you've seen in NumPy
# (as you know it)

# in ML, every dataset is a table/a matrix

# it has a specific row and a column

# normally, if you did:

# np.array([1.0, 2.0])

# this is fine - you'd have data

# but when it comes to ACTUAL MACHINE LEARNING i.e. linear algebra, TF/PyTorch interoperability

# you should ALWAYS have data in rows and columns/table

# so rather than np.array([1.0, 2.0]) which would just be a row vector with some numbers

# and still being data - it isn't in the format they usually come in practical applications - you 

# should actually represent data in their preferred form - a table/matrix

# even if you get X_train in real-life with one feature in it (so only one column) you do not

# represent it in a single row vector with shape (m,) or (only rows,) - columns still EXIST and this 

# is what ML actually needs; real meaning of data i.e. from their table-like shape

# so (m, ) -> convert to 2D -> (m,n) -> get real meaning -> number of examples, number of features

# how many rows does it have - so that I know how many training examples there are

# and how many columns does it have - so that I know the number of features

# you need it to be explicitly evident of the data's structure

# if you needed, say, three features then X_train would be

# X_train = np.array([[1.0, 2.0, 3.0],
#                     [4.0, 5.0, 6.0]]) 

# i.e.

# 3 features, 2 training examples

# Tensor <-> matrix

fig, ax = plt.subplots(1, 1)

ax.scatter(X_train, y_train, marker='x', c='r', label='Data Points')

ax.legend(fontsize='xx-large')

ax.set_ylabel('Price (in 1000s of dollars)', fontsize='xx-large')
ax.set_xlabel('Size (1000 sqft)', fontsize='xx-large')
plt.show()

# lets now create a layer that has 1 linear reg. unit

# a linear reg. unit -> computes a linear reg. output z with linear model (wx + b/wvec*xvec + b)
# and activation is linear / so inherently NO CHANGE/a = z

# a log. reg. unit -> still computes a linear reg. output with linear model (wx + b/wvec*xvec + b) 
# and the activation is sigmoid.
# so, a = sigmoid(z)

linear_layer = tf.keras.layers.Dense(
    units=1,
    activation='linear')

# ONE Dense layer and
# just ONE neuron for now - a linear reg. unit/neuron

w, b = linear_layer.get_weights()

print(w, b)

# now, when you do the 'linear_layer' initialization, it just CREATES 1 layer with 1 lin. reg. 
# unit/neuron and the activation is 'linear'

# but this layer DOES NOT INITIALIZE PARAMETERS YET/INITIALIZE SHAPE OF PARAMETERS YET because

# it DOES NOT KNOW how many input features are coming

# NOTE: Parameters i.e. weights and biases are initialized for each neuron

# by the layer only after the layer sees/infers from data the number of input features

# NOTE: the neuron DOES NOT CARE about the training examples that are incoming

# only no: of features are inferred from data, translating to the no: of parameters to be 

# initialized/shape of parameters to be initialized

# only when the layer/unit SEES the number of features from input data, it can get to know 

# the SHAPE OF the parameters to then initialize the parameters

# So, if input data is something like:

# X_train[0].reshape(1,1) - as seen below

a1 = linear_layer(X_train[0].reshape(1,1))

# when the layer is built for the first time,

# i.e.

# when the layer is passed with this input data,

# it determines how many features the data contains

# which essentially means

# how many features does each example have

# we give it - X_train[0].reshape(1,1)

# so the shape of the input data here is (1,1) -> 1 example, 1 feature
#                                                            ^^^^^^^^
#                                                 this is what the layer understands

# so, feature(s) in each example = 1

# and in layer, we have 1 unit

# so shape of weight W = (1,1) -> (number of features in 1 eg., number of units in layer)

# shape of bias b = (1,) -> number of units in layer

# so the layer:

# -> infers number of features from incoming data
# -> combines it with the number of units it has
# -> to initialzie parameters/shape of parameters

# let's see the random initialization in action:

# so, W = [[w]] -> a single scalar value wrapped into a 2D structure

# b = [b] -> a single scalar bias in 1D

# NOTE: notations, henceforth

# layer 1 -> our layer/linear_layer (not numbered but assume)

# w1_(1) -> weight vector for neuron 1 in layer 1
# w2_(1) -> weight vector for neuron 2 in layer 1
# w3_(1) -> weight vector for neuron 3 in layer 1
# and so on...

# so, in our case, 

# W contains only w1_(1) -> weight vector for neuron 1 in layer 1

# and w1_(1) only contains a single value/scalar value as there is only 1 feature (in each eg.)

# so, W = [w1_(1)] = [[0.73]] <-- randomly initialized

# b = [-0.12] <-- randomly initialized

# Another example

# Say you had 3 features and 2 neurons in layer 1

# so, W shape = (3, 2)

# now, W contains w1_(1), w2_(1); for neuron 1 and 2 in layer 1

# W = [w1_(1), w2_(1)]

# and we have three features;

# this means weight vectors w1_(1) and w2_(1) each have three different values in them
# (hence a 'vector')

# BUT, they are stacked vertically - transposed

# W = [[ | , |] <--- feature 1
#      [ |,  |] <--- feature 2
#      [ |,  |]] <--- feature 3
#        ^   ^
#     Unit1 Unit2
#    w1_(1) w2_(1) 

# And now, values randomly initialized could be something like:

# W = [[ -0.1, 12] 
#      [0.04, 11] 
#      [9.92, -5.67]] 

# NOTE: 'W' is for a SINGLE LAYER; layer 1 in our case

# when the layer has parameters initialized/shape of parameters initialized, we say that the

# layer is 'built'

# you can check that: linear_layer.built
# Output: True -> layer is built | False: layer is not built i.e. parameters haven't been 
# initialized for that layer

# and the layer initially IGNORES the number of examples in incoming/input data into the layer

# NOTE: incoming input -> could be data/activations

# and only cares about the feature count - the last dimension of input/incoming data

# the weights and bias are then initialized for each neuron in layer 1

# in our case - only one neuron

# and only NOW,

# after W - parameters for this/that layer - is initialized

# the layer sees the first example from incoming data

# so, it sees from X_train[0].reshape(1,1) not just the feature count now but the 

# first example from this X_train[0].reshape(1,1) input data

# essentially, from X_train, X_train[0].reshape(1,1) is actually this:

# [[1.0]]

# the layer sees the first example from this input incoming data

# [1.0]

# and now, 

# each neuron in the layer will use its weight/weight vector and bias initialized for it

# i.e.

# in our case, only ONE NEURON and only w1_(1)

# and the first neuron applies it to this example 

# to compute output z

# z = wx + b

# z1_(1) = 0.73(1) + (-0.12)

# z1_(1) = 0.61

# now the neuron computes activation = linear

# z1_(1) = a1_(1) = 0.61

# AND THAT'S IT

# The output of any layer is the output of ALL NEURONS inside the layer

# more specifically,

# the output activation vector of a layer is:

# (number of examples the layer sees from input/incoming data, number of neurons)

# here, the output of layer 1 is the output of ONE NEURON -> has only ONE NEURON (all it has)

# and the number of examples in input data is 1

# so the activation vector, equals a1_(1)

# has shape = (1,1)
#              | |___________________________________________________________   
#  no: of examples layer sees from input/incoming data                      |
#  -> from X_train[0].reshape(1,1)                             number of neurons in layer
#  -> just one example 

# so, a_(1) is just:

#       = [[a1_(1)]]

# with shape = (1,1)

# NOTE: conceptually, we represent/containerize a_(1) as [a1_(1), a2_(1), a3_(1),...] 

# but in TF, you have to actually understand what the shape of a_(1) actually is

# now the next layer will recieve this:

# layer 2 will recieve the activation vector of layer 1 -> a_(1)

# imagine layer 2 has 3 neurons in it

# and all activation = linear

# rather than having a_(1) = [[a1_(1)]] as input from layer 1, for better understanding, we'll use

# a_(1) = [[a1_(1), a2_(1), a3_(1)]] <---------------------------------------------
#                                                                                 |
# so this new, imagined layer that gives this new a_(1) as output has:            |
# 1. 3 neurons                                                                    |
# 2. only one example existed in input/incoming data                              |
#                                                                                 |
# if two examples in input/incoming data, then a_(1) would look something like:   |
#                                                                                 |
# a_(1) = [[a1_(1), a2_(1), a3_(1)], <-- for example 1                            |
#          [a1_(1), a2_(1), a3_(1)]] <-- for example 2                            |
#                                                                                 | 
# so layer 2 recieves this a_(1) activation vector now ----------------------------

# with shape = (1, 3)

# now, the same process happens:

# 1. layer 2 sees the input activation vector
# i.e. a_(1) - output of layer 1 - becomes input of layer 2
# something like layer_2(a_(1)) 

# 2. layer sees the shape of input activation vector
# shape = (1, 3)

# 3. infers that activation count = 3
# inherently, 'activation count' is something like 'feature count'

# and now, parameters are initialized:

# neuron 1 -> w1_(2)
# neuron 2 -> w2_(2)
# neuron 3 -> w3_(2)

# W will now contain = [w1_(2), w2_(2), w3_(2)]

# and the shape of W will be -> (number of activation, number of neurons) -> (3,3)

# b will contain = [b1_(2), b2_(2), b3_(2)]

# so, as we see that the shape is (3,3) we can infer that

# and based on the activation count/like feature count, 

# w1_(2) -> vector will contain 3 values -> [0.17, 8.0, 12.3]
# w2_(2) -> vector will contain 3 values -> [-0.234, 19.6, 0]
# w3_(2) -> vector will contain 3 values -> [9.9, 6.4, -0.16]

#                                            random init.

# now transpose each weight vector (initialized for each neuron)

# so, W will now be:

# W = [[0.17, -0.234, 9.9],
#      [8.0, 19.6, 6.4],
#      [12.3, 9, -0.16]] 

# shape = (3,3) CORRECT

# 4. now output/activation production happens

# a_(1) was = [[a1_(1), a2_(1), a3_(1)]]

# so, we see that our input/incoming activations/activation vector have

# 1 example
# 3 features/activations

# and our layer 2 has three neurons

# so, this means our output activation vector from layer 2 

# a_(2)

# will have a shape of (1, 3)

# 1 -> 1 single example from input activation vector a_(1) from layer 1
# 3 -> 3 neurons in our layer

# so:

# layer 2 sees first example from input activ. vector (only example there is)
# neuron 1 -> computes z1_(2) -> linear activation -> a1_(2)
# neuron 2 -> computes z2_(2) -> linear activation -> a2_(2)
# neuron 3 -> computes z3_(2) -> linear activation -> a3_(2)

# there are no other examples left

# and so the output vector a_(2) would look something like:

# a_(2) = [[a1_(2), a2_(2), a3_(2)]]

# that's it!

# the output is in tf.Tensor, which is the equivalent of np.ndarray or an array datatype but a

# TF array -> Tensor

a1.numpy()

# input/incoming data here is just 1 training example here

# [[1.0]]

# earlier, we gave it the image vector x with 64 values -> this is ONE training example as well

# with 64 features

# [[...]]
#    ^
# 64 values

print(a1)

# NOTE: a1 -> a_(1)

# you are printing the activation of the layer

# this layer has:

# 1 neuron; sees 1 example with 1 feature in it

# so activation vector has shape (1,1)

# (number of examples, number of neurons)

# i.e. [[a1]] <-- for that 1 example

W, b = linear_layer.get_weights()

print(w, b)

# let's set w and b to our own values as its being initialized randomly

set_W = np.array([[200]])
set_b = np.array([100])

linear_layer.set_weights([set_W, set_b])

# NOTE: the first argument in set_weights is the list of parameters: weight array and the bias array

# that's why you enclose the parameter array set_W and set_b in a list

print(linear_layer.get_weights())

# check if parameters have been set

a1 = linear_layer(X_train[0].reshape(1, 1))

# you are reshaping a (1,) - 1D array value - of [1.0] into [[1.0]]

# if you have number of elements = 1, then valid reshapes are (1,1) and (1,)

# you cannot do .reshape(2,1) because you would need 2 elements/values for that

# and 2 x 1 = 2 elements needed from the array (X_train[0] in our case -> 1D array)

# our 1D array only has 1 element -> we need 2 to reshape it -> so you get an error

# elements -> scalar values ONLY

# you might think: in a 3D array, elements are 2D arrays and this count of 2D arrays is what

# corresponds to the elements that you need when multiplying the rows and columns set for reshaping 

# into

# WRONG

# if you had a 3D array like:

# x = np.array([
#     [[1,2],
#      [3,4]],

#     [[5,6],
#      [7,8]]
# ])

# you'd think: no of elements = 2 as there are 2 2D arrays

# but when trying to reshape, you only need to consider the scalar values in the 3D array

# i.e. the single values

# so we have 8 single scalar values

# so valid reshapes are all the rows and columns set that when multiplied together should 

# yield 8

# so, .reshape(4,2), .reshape(2, 4), .reshape(1, 8), .reshape(8, 1) (you get the point)

# even .reshape(2,2,2) -> unless and until produt set rows and columns for reshaping = number of

# scalar values you have

print(a1) # you'll get 300

print(a1.numpy())

a_linear = np.dot(set_W, X_train[0].reshape(1,1)) + set_b
# set_W shape is (1,1)
# X_train[0] shape is (1,) -> [1.0]
# as told above already, you have to reshape it to (1,1) shape; exactly like how set_w is
# so, X_train[0].rehshape(1,1) => [1.0] -> [[1.0]]

# both a1 (a_(1)) and a_linear produces the SAME values

# use our linear layer to make predictions on our training data

X_train_Tensor = tf.convert_to_tensor(X_train)

pred_tf = linear_layer(X_train_Tensor)

# now the input/incoming data to your linear_layer/layer 1 has 2 examples

# so the activation vector produced now will have shape

# (number of examples in incoming/input data, number of neurons)

# so, it will be: (2, 1)

# NOTE: linear_layer/your layer only has ONE neuron

print(pred_tf)

pred_linear = np.dot(X_train, set_W) + set_b

print(pred_linear)

# neuron with sigmoid activation

X_train = np.array([0., 1, 2, 3, 4, 5], dtype=np.float32).reshape(-1,1)  # 2-D Matrix

Y_train = np.array([0,  0, 0, 1, 1, 1], dtype=np.float32).reshape(-1,1)  # 2-D Matrix

# new data - ALWAYS - any DATA - should be in a 2D matrix/table format

pos = Y_train == 1
neg = Y_train == 0

fig, ax = plt.subplots(1, 1, figsize=(4,3))

ax.scatter(X_train[pos], Y_train[pos], marker='x', s=80, c='red', label='y=1')

ax.scatter(X_train[neg], Y_train[neg], marker='o', s=100, label='y=0', facecolors=None, lw=3)

ax.set_ylim(-0.08, 1.1)

ax.set_ylabel('y', fontsize=12)

ax.set_xlabel('x', fontsize=12)

ax.set_title('One variable plot')

ax.legend(fontsize=12)

plt.show()


# so we use Sequential for this

# to string different layers and make a model/network

model = tf.keras.Sequential([
    tf.keras.layers.Dense(
        units=1,
        activation='sigmoid',
        name='L1'
    )
])  

# input_dim = number of features (per training example) that is incoming

# X_train shape = (2,1)

# [
#   [1.0],
#   [2.0]
# ]

# 2 training examples
# 1 feature

# therefore:

# Dense(1, input_dim=1)

# tells TF:

# "every training example reaching this layer
# will have exactly 1 feature"

# so TF knows:

# one neuron needs:
# 1 weight
# 1 bias

# if input_dim = 3:

# Dense(2, input_dim=3)

# TF knows:

# 3 features
# 2 neurons/units

# therefore:

# weights.shape = (3,2)
# biases.shape = (2,)

# we can't give in the 'input_dim' parameter here; some TF/Keras issue that

# could not be resolved

# so the layers won't know how many features it will input

model.summary()

# this shows layers and number of parameters in the model/network

# here: there is 1 layer, and one unit and the unit has one weight w and one bias b

# i.e. W for this layer only contains w1_(1) -> W = [w1_(1)]

logistic_layer = model.get_layer('L1')

print(logistic_layer.get_weights())  

# again, W and b are not yet initialized as logistic_layer.get_weights() is not seeing any

# training example yet - you cannot unpack 0 values into 2 variables/parameters W and b

# like W, b = logistic_layer.get_weights() because

# you're doing W, b = [], which is erroneous

# so if you printed out logistic_layer.get_weights(), you'd just get:

# []

# (like we saw above)

print(logistic_layer.built)

# you'd get False - the model/layer hasn't seen any data yet to initialize weights

# for the version of TF I had on 4/6/26, 6:47 PM -> Version 2.17/Keras 3

# in the Lab, you see 'input_dim=1'

# this 'input_dim=1' is done to specify the number of features the input data has

# i.e. the input data in the input layer 

# in older TF/Keras tutorials, what they do is:

# if you want to create a model/network with 3 layers in it,

# you do:

# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(
#         units=25,
#         input_dim=3,                      < layer 1
#         activation='relu'
#     ),

#     tf.keras.layers.Dense(
#         units=15,
#         activation='relu'                < layer 2
#     ),

#     tf.keras.layers.Dense(
#         units=1,
#         activation='sigmoid'             < layer 3
#     )
# ])

# all will be Dense layers

# input/input layer/input data approximately mean the same thing

# and the first Dense layer - layer 1 - will have this input_dim argument as it 

# behaves as the FIRST layer that retrieves from input/input data/input layer

# and every other layer in the model - layers 2 and 3 - don't need the parameter input_dim because

# TF/Keras already knows that layer 3 will input a vector of activations of size of 

# units/Neurons of layer 2 which we already specified -> 15 units specified for Dense layer 2

# so specifying input_dim in each of the layer would make code redundant - not recommended

# but the layer retrieving from the input data/layer will not know how many features it will be

# getting - it does not have anything before it/before hand to infer from like other layers

# that's why we specify input_dim=1 in the FIRST layer that will retrieve input/input data/input 

# layer 

# the input of this layer is THE INPUT OF THE NEURAL NETWORK - and this FIRST layer would not

# know how many features it will see like other layers

# layer 2 knows it will have as input -> layer 1's output -> an activation vector of 25 features

# layer 3 knows it will have as input -> layer 2's output -> an activation vector of 15 features

# but layer 1 does not know what it will have as input

# that's why we specify input_dim for the FIRST layer that retrieves input/input data/from input

# layer

# model = tf.keras.Sequential([
#     tf.keras.Input(shape=(3,)),
#     tf.keras.layers.Dense(25, activation='relu'),
#     tf.keras.layers.Dense(15, activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

# this is the modern approach - you use Input and not a Dense and specify 'input_dim=1'

# to make it the input layer

# so, if we did it modernly

model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)), # the input layer

    tf.keras.layers.Dense(1, activation='sigmoid', name='L1') # a layer - retrieves from input
])

logistic_layer = model.get_layer('L1')

# this layer DOES NOT KNOW how many features it will input from the input layer

# because its retrieving from the input layer which - itself - does not know how many features

# it will see in the TRAINING EXAMPLE 

# and so if you specify the input's/input layer's/input data's number of features for a sinlge

# training example, the layer that will firstly retrieve from the input/input layer/input data

# will know that 'Oh, I will recieve 1 feature - so I should set one weight w and one bias'

# (our case)

print(logistic_layer.get_weights())

# this is the layer FIRST retrieving from input/input layer/input data

# this layer previously mirrored (in Lab)

# tf.keras.layers.Dense(units=1, input_dim=1, activation='sigmoid', name='L1')

# and this is the layer retrieving from the input/input data/input layer

# and it does not know how many features this input/input data/input layer has

# and so we set input_dim=1

# in the modern approach,

# you don't have to do this - just specify Input and then TF will automatically infer from

# then onwards as we've helped the FIRST layer retrieving from the input/input data/input layer

# with the number of features it will be taking in (in a training example, of couse) so that

# it can set parameter shape and initialize it with values

# now, you see that weights are initialized here, with the desired shape of each parameter

# so, you can now do:

w, b = logistic_layer.get_weights()

print(w, b)

# [[-1.2424643]] [0.]

# so, parameters have been initialized with the

# 1. desired shape to accomodate input features
# 2. randomly initialized value for each parameter

set_w = np.array([[2]])

set_b = np.array([-4.5])

# you can now set these weights yourself (as seen before)

logistic_layer.set_weights([set_w, set_b])

print(logistic_layer.get_weights())

# predict
a1 = model.predict(X_train[0].reshape(1,1))

# you are giving: 1 training example (X_train[0].reshape(1,1))

# has 1 feature

# model has 1 unit

# so returns 1 prediction

# with 1 w and 1 b set by us (after being initialized randomly)

# to accomodate that 1 feature in the training example given

print(a1)

# normal logistic prediction

alog = sigmoid(np.dot(set_w, X_train[0].reshape(1,1)) + set_b)

print(alog)

# both normal and model/network prediction is same