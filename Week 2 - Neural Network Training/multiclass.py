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

print("X_train")
print(X_train[:5])

print("Y_train")
print(y_train[:5])

# see data


plt.figure(figsize=(7, 6))

plt.scatter(
    X_train[:,0],
    X_train[:,1],
    c=y_train,
    cmap='viridis',
    s=60
)

plt.scatter(
    cent[:,0],
    cent[:,1],
    c='red',
    marker='X',
    s=250,
    label='Centers'
)



plt.legend()
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Training Data with Cluster Centers")
plt.show()



# show unique classes
print(f"Unique classes: {np.unique(y_train)}")

# show class representation
print(f"Class repr.: {y_train[:10]}")

# model
tf.random.set_seed(1234)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(2, activation='relu', name='L1'),
    tf.keras.layers.Dense(4, activation='linear', name='L2')
])

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.01)
)

model.fit(X_train, y_train, epochs=10)

# layers

l1 = model.get_layer('L1')
W1, b1 = l1.get_weights()

# these are W1_(1) and b1_(1)

l2 = model.get_layer('L2')
W2, b2 = l2.get_weights()

# create the 'new features', the training examples after L1 transformation
Xl2 = np.maximum(0, np.dot(X_train,W1) + b1)

# done