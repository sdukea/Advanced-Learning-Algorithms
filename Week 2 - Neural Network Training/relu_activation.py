import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LeakyReLU
from tensorflow.python.keras.activations import linear, relu, sigmoid

from matplotlib.widgets import Slider
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

from lab_coffee_utils import load_coffee_data

X, Y = load_coffee_data()

print(X.shape, Y.shape)

# X -> (200, 2)
# Y -> (200, 1)

# 200 examples | 2 features
# 200 binary labels | 1 column

# normalize data

print("Pre-normalization:")
print(f"Temperature; max, min: {np.max(X[:, 0])},{np.min(X[:, 0])}")
print(f"Duration; max, min: {np.max(X[:, 1])}, {np.min(X[:, 1])}")

