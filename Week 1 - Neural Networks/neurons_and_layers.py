# this is how a neural network works
# say we're trying to build a neural network that predicts whether an input image is 0 or 1

# image size: 8 x 8 pixels
# input layer - image vector x (8 x 8 = 64-values)
# first hidden layer - 25 neurons/units
# second hidden layer - 15 neurons/units
# final output layer - 1 neuron/unit (scalar)

# say our activation function is a sigmoid function

# the first hidden layer has 25 neurons/units
# and each neuron sees image vector x

# now, each neuron is also assigned a set of parameters w1 to w64 and b
