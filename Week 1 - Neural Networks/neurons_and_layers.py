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
# you see ROWS now as training examples - one single feature in our case

# X_train is of shape (2,1) now

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

# you should ALWAYS have data in rows and columns

# so rather than np.array([1.0, 2.0]) which would just be a row vector with some numbers

# and it still is data - not in the format they usually come in practical applications - you should

# actually represent data in 2D matrices; like a table

# even if you get X_train in real-life with one feature in it (so only one column) you do not

# represent it in a single row vector with shape (m,) or (only rows,) - columns still EXIST and this 

# is what ML actually needs; real meaning of data

# how many rows does it have - so that I know how many training examples there are

# and how many columns does it have - so that I know the number of features

# you need it to be explicitly evident of the data's structure

# if you needed, say, three features then X_train would be

# X_train = np.array([[1.0, 2.0, 3.0],
#                     [4.0, 5.0, 6.0]]) i.e.

# 3 features, 2 training examples

# Tensor <-> matrix

fig, ax = plt.subplots(1, 1)

ax.scatter(X_train, y_train, marker='x', c='r', label='Data Points')

ax.legend(fontsize='xx-large')

ax.set_ylabel('Price (in 1000s of dollars)', fontsize='xx-large')
ax.set_xlabel('Size (1000 sqft)', fontsize='xx-large')
plt.show()

# lets now create a layer that has 1 linear reg. unit

# a linear reg. unit -> computes a linear reg. output z with linear model (wx + b/wvec * vec + b)
# and activation is linear / so inherently NO CHANGE/a = z

# a log. reg. unit -> still computes a linear reg. output with linear model (wx + b/wvec*xvec + b) 
# and the activation is sigmoid.
# so, a = sigmoid(z)

linear_layer = tf.keras.layers.Dense(
    units=1,
    activation='linear')

# one layer and
# just one neuron for now - a linear reg. unit

# w, b = linear_layer.get_weights()

# print(w, b)

# now, when you do the 'linear_layer' initialization, it just CREATES 1 layer with 1 lin. reg. 
# unit and the activation is linear

# but it does not RANDOMLY initialize weights yet

# only when the layer/unit SEES tr. eg./data, it will randomly assign weights to it

# So,

a1 = linear_layer(X_train[0].reshape(1,1))

# the output is in tf.Tensor, which is the equivalent of np.ndarray or an array datatype but a

# TF array -> Tensor

a1.numpy()

# give it ONE training example (with 1 feature)

# earlier, we gave it the image vector x with 64 values -> this is ONE training example with 64

# features

# don't confuse - essentially, each neuron sees one training example and then randomly assigns

# weights (and bias) to each feature

# in our case, 1 training example has 1 feature

# so one weight (and one bias)

print(a1)

# you're printing the activation = the linear reg. output

# now, you could see how the randomly initialized w and b i.e. the preferred weight/parameter

# pattern matched with the input's feature (only one feature) pattern

w, b = linear_layer.get_weights()

print(w, b)

# so, that 'w' * 1.0 + b = z1 = a1

# this is THAT RANDOMLY INITIALIZED 'w' and 'b' for this neuron that sees that one training 

# example with one feature (one w and one b), just like how each neuron sees one image/training 

# example with 64 features and assigns randomly initialized 64 weights to it (and one bias)

# each time, 'w' and 'b' vary - randomly initialized

# let's set w and b to our own values as its being initialized randomly

set_w = np.array([[200]])
set_b = np.array([100])

linear_layer.set_weights([set_w, set_b])

# NOTE: the first argument in set_weights is the list of parameters: weight array and the bias array

# that's why you enclose the parameter array set_w and set_b in a list

print(linear_layer.get_weights())

# what you're doing here: "Instead of random parameters, let's manually choose parameters whose 
# behavior we understand."

# so now, the value of a1, which was always changing due to w and b being initialized randomly,

# is always: a1 = 200(1) + 100 = 300

# now this is for ONE NEURON -> 1 FEATURE -> 1 TR. EG.
#                                          w = [[200]], b = [[100]]

# if you had three features where X_train looked like:

# X_train = np.array([[1.0, 2.0, 3.0],
#                    [4.0, 5.0, 6.0],
#                    [7.0, 8.0, 9.0]])

# so now, for ONE NEURON -> 3 FEATURES -> 1 TR. EG.
#                                         w should be = [[w1_1], <--- feature 1
#                                                        [w2_1], <--- feature 2
#                                                        [w3_1]] <--- feature 3
#                                                          ^
#                                                       Neuron 1

# why: because shape of w should ALWAYS be in a table/2D form

# w.shape = (3,1) now -> TF now naturally knows that we have 3 features and we have 1 neuron

# shape of data = (number of training examples, number of features)
#               = (3,1)

# weight w shape = (number of features, number of neurons/units)
#               = (3, whatever the number of neurons are)

# Yes, it does not mean the size of training data matrix and the size of weight matrix
# (mentioned 'matrix' because in TF implementation, all data is a matrix of 2D shape) should
# always match - stop being dumb!

# for TWO NEURONS -> 3 FEATURES -> each NEURON SEES 1 TR. EG.

# w should be:
#              = [[w1_1_val_1, w2_1_val_1], <--- for feature 1
#                 [w1_1_val_2, w2_1_val_2], <--- for feature 2
#                 [w1_1_val_3, w3_1_val_3]] <--- for feature 3
#                      ^           ^
#                  Neuron 1     Neuron 2

# So if you want to modify/set weights w for when your dataset has 3 features and 2 neurons, then

# your w array should look like above

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

a_linear = np.dot(set_w, X_train[0].reshape(1,1)) + set_b
# set_w shape is (1,1)
# X_train[0] shape is (1,) -> [1.0]
# as told above already, you have to reshape it to (1,1) shape; exactly like how set_w is
# so, X_train[0].rehshape(1,1) => [1.0] -> [[1.0]]

# both a1 and a_linear produces the SAME values

# use our linear layer to make predictions on our training data

X_train_Tensor = tf.convert_to_tensor(X_train)

pred_tf = linear_layer(X_train_Tensor)

# now pred_tf here is predicting result on our entire dataset 

# i.e.

# you have one neuron and for this neuron, for each tr. eg., you return a prediction with the

# linear model that has w = set_w and b = set_b

# so, inherently,

# TF sees from your dataset TWO rows/TWO in its 2D shape of (2,1)/TWO tr. eg.

# and only for training examples each, you predict

# using the set parameters/rand. initialized parameters for each feature in each TR. EG./ROW

# which is sees again from X_train shape of (2,1)
#                                              ^

# ONE/1 feature in our case
# |
# so one weight w, and one b (either initialized/set randomly)

# for each ROW/TR. EG. now, the layer returns a prediction -> EACH NEURON in layer returns one 

# prediction one Tensor consisting of predictions of each tr. eg. in the input dataset

# So, pred_tf:

# [[300.]
#  [500.]], shape=(2, 1), dtype=float32)

# in layer: 1 Neuron/unit -> sees dataset of two training examples and 1 feature -> initializes one

# weight and one bias (sets/random) for each tr. eg. as each tr. eg. has 1 feature only -> gives 2 

# predictions in total i.e. 

# (the usual linear model pred. explanation follows here)

# 1 for each each training example and 

# each prediction is the linear model w/ attached set weight and bias for how many ever features/
# or feature values (1 in our case) that

# exist in the tr. eg./tr. eg. vector (if more than one feature) and by attaching the weights to 

# the tr. eg./tr. eg. vector feature values, you get the prediction for that example


# NOTE: linear_layer/your layer only has ONE neuron

print(pred_tf)

pred_linear = np.dot(X_train, set_w) + set_b

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

# before, we build a single layer - linear_layer - with one neuron in it

# this ONE neuron sees a training example, uses our set_w and set_b as the set weights for this

# layer i.e. it uses these weights to apply to training examples and each training example

# gets a prediction -> training eg. - apply set weights set_w and set_b to linear model with

# input training example to get z value as well as z = activation a value 

# (as this is a linear reg. unit and the output of each neuron equals the activation as well as 

# the activation here is linear)

# so, we did:

# input tr. eg. -> z = wx + b/wvec * xvec + b -> linear activation/z = a -> output is a

# we just studied what a layer is, and what a neuron could do

# So, you can see that there is no network here, or a model here

# where this one layer exists and one neuron exists inside this layer - 

# it is not defined INSIDE some neural network so that we can call this a neural network

# with a single layer (linear layer) and a single linear reg. unit

# but let's now connect multiple layers, each with any number of neurons

# this is what we call a model/network -> a collection of layers/neurons

# a model: 25 neurons -> 15 neurons -> 1 neuron
#           layer 1        layer 2      layer 3


# so we use Sequential for this

model = tf.keras.Sequential([
    tf.keras.layers.Dense(
        units=1,
        activation='sigmoid',
        name='L1'
    )
])  

# input_dim = number of features per training example

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

# neuron needs:
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

logistic_layer = model.get_layer('L1')

print(logistic_layer.get_weights())  

# again, w and b are not yet initialized as logistic_layer.get_weights() is not seeing any

# training example yet - you cannot unpack 0 values into 2 variables/parameters w and b

# like w, b = logistic_layer.get_weights() because

# you're doing w, b = [], which is erroneous

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

# all Dense layers

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