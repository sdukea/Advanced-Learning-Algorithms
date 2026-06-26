import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
import tensorflow as tf

# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
# from tensorflow.python.keras.losses import SparseCategoricalCrossentropy

from sklearn.datasets import make_blobs

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

from typing import cast

def my_softmax(z): # z -> an array of all z-values from neurons/units
    ez = np.exp(z)              # ez -> an array where each z-value is exponentiated with base 'e'
    sm = ez/np.sum(ez)          # sm -> an array where the 'ez' array is divided i.e. each value
                                # in the array is divided with the value of np.sum(ez) which is a
                                # single value
    return(sm)


# make datasets
cent = np.array([[-5, 2], [-2, -2], [1, 2], [5, -2]])

X_train, y_train = cast(
    tuple[np.ndarray, np.ndarray],
    make_blobs(
        n_samples=2000,
        centers=cent,
        cluster_std=1.0,
        random_state=30
    )
)

print(X_train, y_train)

# X_train -> (2000, 2)
# has 2000 examples, each containing 2 features

# multiclass - one example is attributed to only ONE class out of a collection of many classes
# multilabel - one example is attributed can belong to multiple classes simultaneously
# NOTE: in multilabel classification, say photo tagging:
# if you have 3 classes: beach, sunset, people
# then, an example could have all these classes simultaneously i.e.
# one image could have true/be tagged beach, sunset and people (if present in the image/hence tagged)
# another image could have true/be tagged with only beach and people and probably no sunset is
# present in the image
# |
# So the image need not have all classes true but it can have them simultaneously true

# NOTE: data in Lab file is used alongisde a helper function to visualize - won't be doing it here

# Softmax - used for multiclass
# So, each example can only belong to ONE class out of the many classes we possibly have/can 
# attribute to

# The 'obvious'

model = tf.keras.Sequential([
    tf.keras.layers.Dense(25, activation='relu'),
    tf.keras.layers.Dense(15, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

# There are four classes in our case we initialize

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
)

model.fit(X_train, y_train, epochs=10)

# working

p = model.predict(X_train)

# p -> (2000, 4)
# each prediction is a probability vector with 4 probabilities that add up to 1 and
# the highest of the probabilities are attributed as the preferred class

print(p[:2])
print("largest value", np.max(p), "smallest value", np.min(p))

# preferred implementation