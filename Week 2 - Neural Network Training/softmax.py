import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
import tensorflow as tf

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.datasets import make_blobs

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

def my_softmax(z): # z -> an array of all z-values from neurons/units
    ez = np.exp(z)              # ez -> an array where each z-value is exponentiated with base 'e'
    sm = ez/np.sum(ez)          # sm -> an array where the 'ez' array is divided i.e. each value
                                # in the array is divided with the value of np.sum(ez) which is a
                                # single value
    return(sm)


# make datasets
cent = np.array([[-5, 2], [-2, -2], [1, 2], [5, -2]])

data = make_blobs(n_samples=2000, centers=cent, cluster_std=1.0, random_state=30,
                              return_centers=False)

print(type(data))