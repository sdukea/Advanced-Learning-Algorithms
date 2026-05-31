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

X_train = np.array([[1.0], [2.0]], dtype=np.float32)
y_train = np.array([[300.0], [500.0]], dtype=np.float32)

# data in TF is represented in a different way than what you've seen in NumPy
# (as you know it)

# in ML, every dataset is a table/a matrix

# it has a specific row and a column

# normally, if you did:

# np.array([1.0, 2.0])

# this is fine as well - you'd have data

# but when it comes to ACTUAL MACHINE LEARNING i.e. linear algebra, TF/PyTorch interoperability

# you should ALWAYS have data in rows and columns

# so rather than np.array([1.0, 2.0]) which would just be a row vector with some numbers

# and it still is data - not in the format they usually come in practical applications - you should

# actually represent data in 2D matrices; like a table

# even if you get X_train in real-life with one feature in it (so only one column) you do not

# represent it in a single row vector with shape (m,) or (only rows,) - columns still EXIST and this 

# is what ML actually needs; what kind of data it is?

# how many rows does it have - so that I know how many training examples there are

# and how many columns does it have - so that I know the number of features

# you need it to be explicitly evident of the data's structure

# Tensor <-> matrix

fig, ax = plt.subplots(1, 1)

ax.scatter(X_train, y_train, marker='x', c='r', label='Data Points')

ax.legend(fontsize='xx-large')

ax.set_ylabel('Price (in 1000s of dollars)', fontsize='xx-large')
ax.set_xlabel('Size (1000 sqft)', fontsize='xx-large')
plt.show()

