import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)

from sklearn.datasets import make_blobs
import tensorflow as tf

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)


