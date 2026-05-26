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

# now, each neuron is also assigned a set of parameters w1 to w64 and b RANDOMLY
# neuron 1 (from h.l. 1) might have a set of parameters w1..w64 and b while neuron 2 also
# has a set of parameters w1...w64 but different ones - randomly assigned to each neuron
# why?
# say you have 25 detectives trying to solve a murder case
# each detectives sees the murder - but ALL OF THEM do not look for the same evidences/answers -
# they have different viewpoints/way of thinking.
# one detective might look for fingerprints, while another looks at blood and another might decode
# footprints/travel and so on

# similarly, every neuron sees the SAME input image vector x1...x64
# but because every neuron starts with DIFFERENT random weights (for each pixel in image)
# each neuron initially "cares" about different parts/patterns of the image

# because weights assigned to each feature value/pixel intensity value determines that feature's
# importance (pixel's importance)
# randomly, one neuron might have larger weights for pixels 23, 45, 2, 6, 63 and 5 (example)
# and this neuron seems to 'care' about this pixel pattern
# we don't know, for each neuron, what pixel pattern it cares about because the weights are 
# randomly generated and assigned (for each neuron)

# one neuron may be very good at detecting lines maybe - another may detect horizontal lines and
# another might detect loops
# and all of this depends on the random initialization of parameters and their weights for each
# neuron - because its random for each neuron, they specialize in different pixel patterns

# so, each neuron outputs ONE value corresponding to:
# how much does this image (vector x) match this MY pixel preference pattern?
# say image vector x is an image 1 – somewhere around the centre of the image, the pixel values/
# features all have a value of 1
# say features x4, x12, x20, x28, x36, x44, x64 are all valued 1
# now say neuron 3 has been randomly assigned, by chance, larger weights for most of these
# features – every feature except x36 has been assigned a larger weight value
# |
# this means the neuron's preferred pattern/internal scoring template matches the image's
# neuron 1 says "I like most of the centre-column pixels to be bright"
# and the image also seems to have bright/1 for all of the centre-column pixels
# let's think term by term

# z3_1 = w1x1 + w2x2 + ... + w4x4 + .. w4x12 + .. w4x20 + ... w64x64
# 

# z1_1 -> neuron 1
# z2_1 -> neuron 2
# z3_1 -> neuron 3
# ....
# z25_1 -> neuron 25

# this output is that ONE value for each neuron that answers the question (line 42)