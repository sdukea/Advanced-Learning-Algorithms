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
# say you have 25 detectives trying to solve a murder case and task is: predict if crimial is
# a PROFESSIONAL or NOT
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
# so each senior detective has their own feature preference/activation pattern now – because they 
# are now looking at a more richer set of information now

# h.l. 1 -> sees pixel intensity values -> owns a specific pixel preference pattern per neuron
# h.l.2 -> sees feature activations -> owns a specific feature activation pattern per neuron

# so, h.l. 1 outputs activation vector a_1, consisting of 25 values/activations

# each senior detective neuron in the second hidden layer does not look at pixel values now;
# they are looking at strenghts of detected low-level patterns/evidences from previous layer/
# junior detectives

# just like how hidden layer 1 sees the strenghts of pixels, hidden layer 2 sees strenght of
# low-level patterns (edges/low-level murder evidences such as fingerprints and so on)
# |
# hidden layer 2 sees low-level FEATURES - the input to hidden layer 2 are features

# for h.l.2, activations from layer 1 are its features
# for h.l.1, pixel intensity values from input vector x are its features (called it pixels before)

# again, each neuron in h.l.2 is randomly assigned weights
# neuron 1 in h.l.2 might recieve a random collection of weights w1 to w25
# (you have 25 weights because the input activation vector has 25 values - one for each of the
# 25 neurons neuron in the previous h.l.1)
# neuron 2 might recieve a random collection of weights w1 to w25
# and so on for the other neurons (15 in total) the h.l.2

# and just like before, each neuron has a specific activation preference now

# say, a_1 =[0.95,0.10,0.87,0.98,…]
# 25 values here

# now, neuron 1 in h.l. 2 might be randomly assigned with a higher weight for the activation
# feature 1, 3, 4, 23, and 17

# from activation vector, consisting of low-level features, we see that
# there is a higher strenghth of 0.95 for feature 1/some kind of edge detector

# and even a higher strength of 0.87 for feature 3/some other kind of edge detector

# and even a higher strengh of 0.98 for feature 4/some other kind of edge detector

# so, neuron 1's activation/strength of low-level feature preference matches MOST of the incoming 
# strenghts of the input activation vector (from h.l.1) as well

# this means neuron 1 activates strongly - will generate a higher value
# (if there are more weight-feature matches in total – in the total of z1_2)
# i.e. THAT value which corresponds to the question in line 42:
# how much does the activations/low-level feature detections match MY preference?

# and this repeats for every neuron in hl 2

# and each neuron activates strongly/weakly based on match between the weights assigned to it
# randomly and the feature strength values from previous hl 1)

# and the output of this hl 2 => a_2

# this is the activation vector that is MORE RICHER than the info/activation vector of a1

# because hl 2 now combines LOW-LEVEL FEATURES as input to detect SHAPES/LARGER CONCEPTS

# if a neuron in the hl 2 prefers:
# vertical edge -> high/strong weight/importance
# slant edge -> low/smaller weight/importance
# loop edge -> high/strong weight/importance
# horizontal edge -> zero/no weight/importance

# and if it matches the input EDGES/LOW-LEVEL FEATURES, then

# this neuron in the hl 2 is actually trying to combine EDGES to see SHAPES

# if junior detectives give reports/strenghts of their clues/evidences/features i.e.
# detective 1: (neuron 1)
# looks for fingerprints (pixel preference pattern - vertical edges maybe)
# returns report (evaluation of pattern and image vector x)

# detective 2: (neuron 2)
# looks for blood stains
# returns report (evaluation of pattern and image vector x)

# detective 3: (neuron 3)
# looks for shoe prints
# returns report (evaluation of pattern and image vector x)

# detective 4: (neuron 4)
# looks for broken windows
# returns report (evaluation of pattern and image vector x)

# detective 5: (neuron 5)
# looks for tire marks
# returns report (evaluation of pattern and image vector x)

# then hl 2 recieves these reports/activation strenghts of each clue/feature like:
# fingerprints strongly detected -> 0.95
# blood weakly detected -> 0.10
# shoe prints strongly detected -> 0.91

# and they make RICHER reports
# i.e.

# senior detectives detect COMBINATIONS of CLUES/EVIDENCE/FEATURES produced by the prev. hl 1 
# of junior detectives

# if for sen. det. 1:
# fingerprints -> strong
# shoe prints -> strong
# broken window -> strong
# via weights assigned (15 of these)

# then senior detective may conclude:
# "this resembles a burglary-crime-like" -> RICH info/high-level feature

# senior detective 2 may think:

# if:
# blood stains -> strong
# no forced entry -> strong
# via weights assigned (15 of these)

# then:
# "this resembles a domestic-crime behaviour" -> RICK info/high-level feature

# and so on for the 15 other senior detectives

# and activation feat. a_2 that has more RICHER information/HIGH-LEVEL structures/features

# Next, the output layer recieves all these 15 RICHER activations/RICHER info

# and random weights are assigned for this as well

# this output layer consists of one neuron

# this neuron is like the chief investigator/detective

# Suppose the chief investigator receives these reports:

# Evidence Pattern (or)
# high-level features	        Strength (from prev. layer)        Weight (initialized now)
# organized behaviour           0.95                               +8
# panic-driven behaviour	    0.40                               -8
# careful-cleanup behaviour     0.91                               +6
# burglary-crime-like           0.10                               +4
# domestic-crime-behaviour      0.05                               -5
# ...

# image 15 of these high level features – strenghts/feature activation and their weights

# now, this chief detective has his/her own pattern of concluding/finalizing/predicting whether
# this was a PROFESSIONAL criminal or NOT

# now, NOTE (analogy from 3b1b neural network video):
# the layer BEFORE the output layer outputs the HIGHEST LEVEL OF FEATURE INFORMATION/ACTIVATIONS in
# that neural network - it is the richest of information

# so the output layer DOES NOT PRODUCE MORE RICHER INFORMATION because we've already closed the
# loop on processing the information required to NOW PRODUCE an output/prediction

# and so the output layer only does the OUTPUT PRODUCTION part - predicting whether the image
# is zero or 1 or predicting whether this is a PROFESSIONAL criminal or NOT/AMATEUR

# because the activations from the previous layer - the hidden layer 2 - is already the highest
# level of features/information/activations we could possibly have

# that's why hidden layer 2 already sees patterns such as burglary-crime-like, organized
# behaviour and so on...
# these high-level features ARE THE HIGHEST LEVEL OF INFORMATION the neural network can predict
# by being before the output layer -
# and you can see for yourself as well – we've already reached features that seem to describe the
# exact crime

# and now, we can only expect to predict the output - not produce an even more richer set of 
# features 

# our only concern is the scalar output (in our case) = z1_3

# now, we know that the output layer recieves 15 activations, each corresponding to a higher/highest
# feature activation that gets us the closest to now predicting whether the criminal is
# PROFESSIONAL or NOT/AMATEUR

# NOTE: These activations are NOT final decisions.

# they are just strenghts of detected high-level behavioural patterns/features

# the output neuron thinks:
# “Given all these behavioral patterns,
# how likely is this criminal professional?”

# and produce ONE VALUE/PROBABILITY/OUTPUT/PREDICTIION (in our case)
# that's it!

# now, this chief detective has his/her own pattern/preference in determining, from all the
# behaviour evidence/activations, whether this is likely a PROFESSIONAL or AMATEUR

# because that's THE ONLY JOB OF THIS NEURON – to predict
# that's it!

# now, at this stage, you're trying to make a normal classification prediction

# you're assigning weights (15 of them; randomly) to certain features (15 of them from prev. layer) 
# you have (got as input)

