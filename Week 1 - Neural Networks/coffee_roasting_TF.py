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