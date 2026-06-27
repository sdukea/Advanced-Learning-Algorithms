import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)

from sklearn.datasets import make_blobs
import tensorflow as tf

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

from typing import cast


# 4-class dataset for classification
classes = 4
m = 100
cent = np.array([[-5, 2], [-2, -2], [1, 2], [5, -2]])
std = 1.0

X_train, y_train = cast(tuple[np.ndarray, np.ndarray], make_blobs(n_samples=m,
                                                             centers=cent,
                                                             cluster_std=std,
                                                             random_state=30))


print(X_train.shape, y_train.shape)

