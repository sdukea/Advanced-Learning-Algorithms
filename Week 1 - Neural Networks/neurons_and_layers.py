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

# because weights assigned to each pixel intensity value determines that pixel's
# importance (pixel's importance)
# randomly, one neuron might have larger weights for pixels 23, 45, 2, 6, 63 and 5 (example)
# and this neuron seems to 'care' about this pixel pattern
# we don't know, for each neuron, what pixel pattern it cares about because the weights are 
# randomly generated and assigned (for each neuron)

# one neuron may be very good at detecting lines maybe - another may detect horizontal lines and
# another might detect loops
# and all of this depends on the random initialization of parameters/weights and their size for each
# neuron - because its random for each neuron and they specialize in different pixel patterns/roles

# so, each neuron outputs ONE value corresponding to:
# how much does this image (vector x) match this MY pixel preference pattern?
# say image vector x is an image 1 – somewhere around the centre of the image, the pixel values all 
# have a value of 1
# say pixels x4, x12, x20, x28, x36, x44, x64 are all valued 1
# now say neuron 3 has been randomly assigned, by chance, larger weights for most of these
# pixels – every pixel except x36 has been assigned a larger weight value
# |
# this means the neuron's preferred pattern/internal scoring template matches the image's
# neuron 1 says "I like most of the centre-column pixels to be bright"
# and the image also seems to have bright/1 for all of the centre-column pixels

# z3_1 = w1x1 + w2x2 + ... + w4x4 + .. w4x12 + .. w4x20 + ... w64x64
# the pixel values in each of the terms either contribute postively or negatively
# i.e. if pixel is valued higher/bright, and if the weight is also higher/gives larger importance
# then that term's value will be larger
# like neuron 3 seems to be assigned with larger weights/higher importance for pixels x4, x12, x20, 
# x28, x44 and x64 whose values are also bright/higher
# (+large)*(+large) = higher value for this term
#   weight  pixel

# and if, for one neuron, a lot of the weight-pixel terms seems to be larger/matches
# pattern/higher, then the value z for that neuron will be larger
# signifying the fact that this image vector x matches the pixel preference pattern/model of this 
# neuron

# z1_1 -> neuron 1
# z2_1 -> neuron 2
# z3_1 -> neuron 3
# ....
# z25_1 -> neuron 25

# this output is that ONE value for each neuron that answers the question (line 42)
# if its high, the pixel preference pattern seems to match the input (vector image x)
# if its low, low match between pattern and input

# now, you apply an activation function – a sigmoid function (in this case)
# to squish the value of z (which can be very high positives/high negatives) between 0 and 1
# this is activation level

# now, take the second hidden layer with 15 neurons
# it does not look at the image vector x/pixel intensity values anymore
# it inputs the activation vector, consisting of 25 different activations, from hidden layer 1

# to understand this intuitively:
# the first hidden layer consists of 25 different junior detectives
# each jr. detective looks at the pixels and with its own pixel preference pattern/role of investi-
# gation (neuron 1/jr. detective 1 might look at fingerprints/this kind of edge and neuron 2/jr.
# detective 2 might look at forensics/this other kind of edge and so on for other 25 detect./neur.)

# now the first hidden layer passes RICHER information to the second hidden layer
# it does not see pixel intensity values anymore. instead, they listen to the junior detective
# scores on each part of the investigation/pixel pattern validation with image

# hidden layer 2 operates on a higher level
# BASED ON THE INCOMING RICHER INFORMATION, these neurons in the hidden layer form opinions
# themselves

# we know that the pixel preferences of each neuron in hidden layer 1 are some kind of features
# in the image – vertical edges, horizontal edges and so on

# and it seems that hidden layer 1 sees EDGES - each neuron in layer 1 is an edge detector
# neuron 1, based on its pixel preference pattern, may detect vertical edges
# neuron 2 might detect slanting edges based on its pixel prefernce pattern
# but neurons in the first hidden layer are basic and simple – juniors

# the second hidden layer are now senior detectives
# they don't care about 64 pixel intensity values now – only the 25 features/pixel preference 
# patterns given by the previous hidden layer 1 by the 25 neurons/jr. detectives

# so, again, weights/parameters are initialized randomly for hidden layer 2 as well
# so each senior detective has their own feature preference pattern now – because they are now
# looking at a more richer set of information now

# so neuron 1/senior det. 1 (out of 15 of them in h.l. 2) might say:
