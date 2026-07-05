import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
plt.style.use('./deeplearning.mplstyle')

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Input, Dense

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

tf.autograph.set_verbosity(0)
# means: don't print AutoGraph informational messages

# You'll see fewer messages such as:

# AutoGraph could not transform ...
# Converting function ...
# Generated TensorFlow graph ...

# load data:

from lab_coffee_utils import load_coffee_data

X, Y = load_coffee_data()

print(X.shape, Y.shape)

# print(X)
# shape: (200, 2)

# so you have 200 training examples, each with 2 features - temperature of coffee beans (in celsius) 
# and duration of roasting in minutes

# print(Y)
# shape: (200,1)

# this is for each tr. eg.: 1 for good roast, 0 for bad roast

# let's plot data

temp = X[:, 0]

duration = X[:, 1]

Y = Y.ravel()

# you have a column vector of labels Y; 2D shape

# i.e. you have it like:

# # array([
#     [1],
#     [0],
#     [1],
#     ...
# ])

# and when you try to do the below: boolean masking

# NumPy expects a 1D array of labels, not a column vector that is 2D

# so, you ravel/flatten it to a 1D array

# so, you turn (200,1) to (200,)

# now, Y == 1 (= good) is now a 1D array that serves as the perfect mask

# why:

# BECAUSE 'temp' is a 1D array 

# and so is 'duration'

# you can only have as mask a 1D array if the data you're applying the mask on is ALSO a 1D array

# so as we have all data (that mask is applied on; temp and duration) as 1D array

# masks that are applied on this should also be of 1D array

# .ravel() is a great function to ravel/flatten a column vector as such to a 1D array

# NOTE: you can also have

# 1. a 2D array with a 1D mask

A = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])

mask = np.array([True, False, True])

A[mask]

# the result:
# array([
#     [1, 2],
#     [5, 6]
# ])

# but what's incompatible is

# A.shape    == (3, 2)
# mask.shape == (3, 1)

# NumPy cannot interpret a (3,1) boolean mask for indexing a (3,2) array
# (think about it/TBI)

# A[mask] 

good = Y == 1
bad = Y == 0

fig, ax = plt.subplots(1, 1, figsize=(1,3))

# good roast
plt.scatter(temp[good], duration[good], marker='x', c='r', s=80, label='Good roast')

# bad roast
plt.scatter(temp[bad], duration[bad], marker='o', s=80, label='Bad roast')

plt.xlabel('Temperature (Celsius)')
plt.ylabel('Duration (minutes)')
plt.title('Coffee roasting')
plt.legend()

plt.show()

# unravel Y
Y = Y.reshape(-1, 1)

# because the shape has to be a 2D array again
# and specifically, has to be (200, 1) again

# normalize data

# fitting weights to the data will proceed more quickly if the data is normalized

print(f"Temperature Max, Min pre normalization: {np.max(X[:,0]):0.2f}, {np.min(X[:,0]):0.2f}")
print(f"Duration    Max, Min pre normalization: {np.max(X[:,1]):0.2f}, {np.min(X[:,1]):0.2f}")

norm_l = tf.keras.layers.Normalization(axis=-1)

# does not normalize anything
# does not compute mean/variance
# does not touch your data

# it creates a MACHINE whose future job is normalization

# its like constructing
# layer = tf.keras.layers.Dense(units=25, activation='sigmoid')
# |
# TF does not immediately compute z = wx + b

# It creates an object that knows how to perform this operation later.

# only if you did: layer(X), you get a prediction/does its job/operation

# the same this is what norm_l/normalization_layer does

# axis=-1

# this says the object that the data you will be seeing - X - has 'features' in the 
# last dimension

# i.e. X dimension -> (200,1)
#                          ^
# and so features are here ^ which is the last dimension

# so axis=-1 tells that the last dimension of my data contains the features

# why these features have to be given:
# "Whenever you learn statistics, learn them separately for each feature." is what you're
# saying to this feature

# so, yes:
# now - for each feature - this object will know to calculate/learn mean and variance
# as you've stated already that you've got 2 features in the data

# so, this object will later operate to learn/calculate:
# mean of temperature feature 
# variance of temperature feature

# mean of duration feature
# variance of duration feature

# or more succinctly

# mean of each feature
# variance of each feature

# buy why not create a function separately that does this normalization?

# Because TF wants this transformation to be part of the model itself

# This can sit inside a network/model like

# model = Sequential([norm_l, Dense(...), Dense(...)])

# and every input can passes through the same learned coordinate transformation/normali-
# zation

# the point of normalization: to know what high, low, normal is for each feature data you have

# coffee might be roasting at 250 degree Celsius, and for about 3 minutes

# is this temperature high or low, and is this time very long or short?

# you need the normal value in each first - the average value/mean

# so that you know such temperature is ABOVE/BELOW AVERAGE temperature

# and such duration is ABOVE/BELOW AVERAGE duration/time

# now you have a more useful system


norm_l.adapt(X) 

# now, norm_l object already knows that it has to operate by learning mean/variance for each feature
# in the incoming dataset, given by the information via the argument 'axis=-1' (last dimension
# of data will hint the feature)

# so, now, the operation BEGINS

# .adapt() lets you calculate/learn the mean and variance of each feature there

# it finds: (just an example)

# temperature_mean = 220
# temperature_std  = 30

# duration_mean = 13
# duration_std  = 1

# and internally, it stores that like:

# mean = [220, 13]
# std  = [30, 1]

# if you did: print(norm_1.adapt(X))

# you'd get None - because learning/calculating these is all INTERNAL

# just like if you did:

# numbers = [2, 2352, 23, 234, 12, 1.8]

# print(numbers.sort())

# you'd get None - even though the number is sorted successfully

# but if you want to see the learned statistics, then you could do:

# print(norm_l.mean.numpy())

# prints all the learned mean for each feature

# print(norm_l.variance.numpy())

# prints all the learned variance for each feature

X_norm = norm_l(X)

# now you feed the entire dataset through the normalization layer

# FIRST, norm_l was made to be a MACHINE whose job was to normalize (like we're doing here -

# normalizing the training data X) + telling it where to look in the dimension of data to 

# know the number of features

# but before creating this object, SECONDLY, you have to make this norm_l learn/calculate the

# mean and variance of incoming features 

# and then LASTLY, after this machine object/layer has learned this, you can now use it to 

# normalize data - layer_object(training_data) -> norm_l(X)

# .adapt(X) is very similar to .fit(X)

# X_norm = norm_l(X) is very similar to X_norm = transform(X)

# and, you know that the normalization formula is:

# X_norm = (X - μ) / σ

# so, for each training example, say you have

# [250, 12]

# where 250 is the temperature and 12 is the duration

# now, the layer remembers:

# Temperature mean = 220
# Temperature std  = 30

# Duration mean = 13
# Duration std  = 1

# so, the new, normalized temp. value for this tr. eg. becomes:

# (250 - 220) / 30 = 1

# and to get the new, normalized duration value for this tr. eg.:

# (12 - 13) / 1 = -1

# X_norm_tr_eg = [1, -1]

# and so on for all the other training examples

print(f"Temperature Max, Min post normalization: {np.max(X_norm[:,0]):0.2f}, {np.min(X_norm[:,0]):0.2f}")
print(f"Duration    Max, Min post normalization: {np.max(X_norm[:,1]):0.2f}, {np.min(X_norm[:,1]):0.2f}")

Xt = np.tile(X_norm, (1000,1))

# how .tile works

# np.tile([1, 2, 3], 3)

# output: [1, 2, 3, 1, 2, 3, 1, 2, 3]

# repeats/duplicates the given array three times

# .tile(X_norm, (1000, 1))

# repeats the first dimension i.e. the rows a 1000 times and 

# the second dimension 1 time

# so prev. dimension: (200, 2)

# new dimension: (200*1000, 2*1) = (200000, 2)

Yt = np.tile(Y, (1000, 1))

print(Xt.shape, Yt.shape)

# tensorflow model

tf.random.set_seed(1234)

# as you know, a layer will be assigned weights randomly (if the weights have not been set)

# now, for every run, weights will still be randomly initialized in the first place (run 1)

# but will stay the same for every run after run 1
# (ykwii)

model = tf.keras.Sequential([
    tf.keras.Input(shape=(2,)),
    tf.keras.layers.Dense(3, activation='sigmoid', name='L1'),
    tf.keras.layers.Dense(1, activation='sigmoid', name='L2')
])

# NOTE:
# the model takes input DATA (a set of EXAMPLES), not a specific EXAMPLE alone

# and for each example in data, it gives a prediction (as you know)

# so model.predict(test_data) will return predictions for each tr. eg. in the input data/test_data

# the 'shape' argument in Input: 

# describes the shape of ONE EXAMPLE, not the entire incoming dataset

# say shape=(2,)

# this actually translates to

# (None, 2) in NumPy/Tensor array of input DATA/test data for prediction

# and as you can see, the '2' here = columns = features

# which matches the meaning of shape=(2,) -> 2 features in each training example

# and inherently, None will specify the NUMBER OF EXAMPLES

# and it can be ANY NUMBER OF TRAINING EXAMPLES

# what shape=(2,) actually says is 'I don't care how many training examples there are –

# I need 2 features in each example'

# so, your input data can be NumPy arrays/tensors of shape

# (32, 2)

# (2, 2)

# (1, 2)

# (10000, 2)

# ...

# say shape=(28,28)

# this now means input data is a 3D array/Tensor 

# and it translates to

# (None, 28, 28) in NumPy/Tensor array

# this means that each training example is 28 x 28 - most likely an image

# and input data/test data for pred. can contain ANY number of eg. of these images and

# their shapes can be:

# (1000, 28, 28)

# (10, 28, 28)

# (2, 28, 28)

# (32, 28, 28)

# ...

# So the shape argument in Input only mentions the number of features in each eg. in Input data

# if shape=(2,) = 1-Dimensional -> Input data SHOULD BE 2-Dimensional only to MATCH
#       ^                                              ^
#     tr. eg.                               input data/NumPy array/Tensor

# why:

# because each training example is known to have 2 features/values (2,) i.e. they are 1D arrays

# so if you are CONTAINERIZING these examples, you can only do so with a 2D array

# so input data/NumPy array/Tensor should be a 2D array containing these 1D examples

# shape=(2,) -> (None, 2)/(32, 2)/(1000, 2)/(10000, 2)... -> MATCHES
#  ^                    ^
# 1D examples   2D input data of 1D examples can be given as input for making predictions for
#               each 1D example now

# if shape=(28,28) = 2-Dimensional -> Input data SHOULD BE 3-Dimensional only to MATCH

# why:

# because each example is known to have 28 x 28 features - a 2D array by themselves

# so if you are CONTAINERIZING these examples, you can only do so with a 3D array

# so input data/NumPy array/Tensor/test data should be a 3D array containing these 2D examples

# shape=(28, 28) -> (None, 28, 28)/(Any number of images, 28, 28)/(1000, 28, 28) ... -> MATCHES
#      ^                                                ^
#   2D image tr. eg.                3D input data/test data containing 2D image
#                                   examples so that we can predict output for each example

print(model.summary())

# the parameters for each layer

L1_num_params = 3 * 2 + 3 
L2_num_params = 1 * 3 + 1 

print(f"Number of parameters:\nLayer 1: {L1_num_params}\nLayer 2: {L2_num_params}")

# FROM HERE ON OUT:

# w1_1 -> layer 1, weight 1
# w2_1 -> layer 2, weight 1
# w3_1 -> layer 3, weight 1

# (changed in 'neurons_and_layers.py' as well)

W1, b1 = model.get_layer('L1').get_weights()
W2, b2 = model.get_layer('L2').get_weights()

print(f"W1{W1.shape}:\n", W1, f"\nb1{b1.shape}:", b1)
print(f"W2{W2.shape}:\n", W2, f"\nb2{b2.shape}:", b2)
# shape of weight parameter initialized: (number of features, number of neurons)
# shape of bias parameter initialized: (number of neurons,)

# compile
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
)

# the compile method sets loss function and the optimization function

# fit
model.fit(
    Xt, Yt,
    epochs=10
)

# know that:
# shape of Xt = (200000, 2)
# shape of Yt = (200000, 1)

# so we have 200000 training examples

# when we do 'epochs=...'

# one complete pass through 200000 training examples = 1 epoch

# epochs = 10; we pass through 200000 training examples 10 seperate times

# it is not:

# 1. see/compute loss - for all 200000 examples
# ------------one epoch done-------------
# 2. optimizing parameters using gradient descent
# 3. repeat this 10 times

# it is actually:

# mini-batch gradient descent

# where each batch size = 32

# what is a batch: a small chunk of the training examples

# and the default size of a batch in TF is about 32 - 32 chunks/batches of the training example

# so, in our case, we have 200000/32 batches = 6250 batches of our training example

# so this is how it actually works:

# 1. see/compute loss - for first batch/first 32 examples (0-32)
# 2. optimize/update parameters via g.descent optimization
# 3. see/compute loss  for the next batch/next set of 32 examples (33-64)
# 4. # 2. optimize/update parameters via g.descent optimization
# 5. see/compute loss - for the next batch/next set of 32 examples (65-96)
# 6. optimize/update parameters via g.descent optimization
# and so on for the 6250 times
# ------------now one epoch is done-------------

# so, for 1 epoch, you get 6250 parameter updates
# for 10 epoch, you get 6250 x 10 = 62500 parameter updates

W1, b1 = model.get_layer('L1').get_weights()
W2, b2 = model.get_layer('L2').get_weights()

print("Updated W1:\n", W1, "\nUpdated b1:", b1)
print("Updated W2:\n", W2, "\nUpdated b2:", b2)

# if you ran this notebook/file more than once,

# you can see that the updated weights are different in each run

# both minimize the cost functions well - they do the task very well - but each run might

# end up leading the parameters to a different valley than the other

# and even the gradients - for each batch - will very minutely differ in every run

# let's now set some weights ourselves

# we're just saying:

# No matter what weights YOUR training produced,
# replace them with/set OUR known weights.

W1 = np.array([
    [-8.94,  0.29, 12.89],
    [-0.17, -7.34, 10.79]] )

b1 = np.array([-9.87, -9.28,  1.01])

W2 = np.array([
    [-31.38],
    [-27.86],
    [-32.79]])
b2 = np.array([15.54])

model.get_layer('L1').set_weights([W1, b1])
model.get_layer('L2').set_weights([W2, b2])

# set custom weights

W1, b1 = model.get_layer('L1').get_weights()
W2, b2 = model.get_layer('L2').get_weights()

# check if the weights have now been set

print("W1:\n", W1, "\nb1:", b1)
print("W2:\n", W2, "\nb2:", b2)

# make predictions now

X_test = np.array(
    [
        [200, 13.9],
        [200, 17]
    ]
)

X_test_norm = norm_l(X_test)

print(X_test_norm.shape)

predictions = model.predict(X_test_norm)

print(f"Predictions:\n{predictions}")

# predictions -> contains predictions for 1 example each

# so predictions returns 2 values - 1 for each training example in test data/X_test_norm

# threshold predictions

yhat = np.zeros_like(predictions)

for i in range(len(predictions)):
    if predictions[i] >= 0.5:
        yhat[i] = 1
    else:
        yhat[i] = 0
print(f"decisions = \n{yhat}")