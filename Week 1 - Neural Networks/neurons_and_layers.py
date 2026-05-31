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

