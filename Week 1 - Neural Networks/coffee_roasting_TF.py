import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

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

# buy why not create a function separately that does this normalization?

# Because TF wants this transformation to be part of the model itself

# This can sit inside a network/model like

# model = Sequential([norm_l, Dense(...), Dense(...)])

# and every input automatically passes through the same learned coordinate transformation/normali-
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

Yt = np.tile(X_norm, (1000, 1))

print(Xt.shape, Yt.shape)

# tensorflow model

tf.random.set_seed(1234)

# as you know, a layer will be assigned weights randomly (if the weights have not been set)

# now, for every run, weights will still be randomly initialized in the first place (run 1)

# but will stay the same for every run after run 1
# (ykwii)

model = Sequential([
    tf.keras.Input(shape=(2,)),
    Dense(3, activation='sigmoid', name='L1'),
    Dense(1, activation='sigmoid', name='L2')
])

# input data/layer is a single training example with 2 features

# and we predict/use the neural network/model on a single training example ONLY

# NOTE: the shape in the Input: specifies the shape of each incoming training example

# i.e.

# inherently the number of features in the incoming training example

print(model.summary())

