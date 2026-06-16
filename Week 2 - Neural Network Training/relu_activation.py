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
