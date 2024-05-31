"""Isaac's Notes on what's required to make a neural network"""

# We will input an n×m matrix with m training examples and n values in each example
# Make it so each column is an example and each row is a feature (still n×m)

# Input layer
# Hidden layers
# Output layer

# Forward propagation (running an input through the network, easy part):
# A_0 will be input layer
# w_1 will be the weights from the input layer to the first layer
# b_1 will be the bias on the first layer
# z_1 is the unactivated first (hidden) layer
# z_1 = w_1 * A_0 + b_1
# ^     ^       ^     ^
# k_1×m k_1×n   n×m   k_1×m
#                   I think b_1 may be considered a k_1×1 vector but in reality it will be
#                   a k_1×m matrix where each column is the exact same k_1×1 vector (the bias)
# where k_1 is the size of the hidden layer (k_1 neurons)

# To find A_1, we apply an activation function to z_1
# We will use ReLU (0 if x ≤ 0 else x)
# A_1 = ReLU(z_1)

# z_2   =   w_2  *  A_1   +   b_2
# ^         ^         ^         ^
# k_2×m     k_2×k_1   k_1×m     k_2×m
# where k_2 is the number of neurons in the second hidden layer

# Once we're done with all the hidden layers, z_L (L for last) will be
# z_L   =   w_L  *  A_(L-1)   +    b_L
# ^         ^             ^          ^
# k_L×m     k_L×k_(L-1)   k_(L-1)×m  k_L×m

# But instead of using ReLU to calculate A_L, we will calculate the output layer
# A_L = softmax(z_L)       # this normalizes the sum of (e^x of each output value) to 1


# Forward propagation done, now time for back propagation (calculus):

# I'm not gonna takes notes on all of this, 3blue1brown explains it better.
# https://www.youtube.com/watch?v=tIeHLnjs5U8&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&t=307s
# Once we've calculated the derivatives,
# we subtract the learning rate times the derivative from every weight and bias.

# useful line:
# print('\n'.join([str(list(item)) for item in one_hot_y]).replace('.0', '').replace('0', ' ').replace(',', '').replace('1', '██'))




