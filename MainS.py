from sklearn.datasets import load_digits
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# digits = load_digits()
#
# # The data and target are already NumPy arrays, but let's explicitly convert them
# data = np.array(digits.data)
# target = np.array(digits.target)
#
# X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=15)
#
# X_train = X_train.T
# X_test = X_test.T
# y_train = y_train  # y's are vectors, no need to transpose.
# y_test = y_test

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        # TODO: return output
        pass

    def backward(self, output_gradient, learning_rate):
        # TODO: update parameters and return input gradient
        pass

class Dense(Layer):
    def __init__(self, input_size, output_size):
        '''output size is the next layer dimension, input size is how much data it is taking in'''
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):

        #find find the change in weights by getting the dot product of the derivative of the output with the tranpose of input
        #after it takes this gradient multiplies by the learning rate and subtracts it from the current weights
        weights_gradient = np.dot(output_gradient, self.input.T)

        #the error with respect to x is dot product of weights transpose by derivaitve of error with respsect to the output
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient

        #change bias is given by the derivative of the output
        self.bias -= learning_rate * output_gradient

        return input_gradient

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))



class Leaky_ReLU(Activation):
    def __init__(self, a=0.01):
        self.a = a
        lr = lambda x: np.maximum(self.a * x, x)
        lr_prime = lambda x: np.where(x > 0, 1, np.where(x < 0, self.a, 0))
        super().__init__(lr, lr_prime)



def get_predictions(A2):
    return np.argmax(A2, 0)


def get_accuracy(predictions, y):
    return np.sum(predictions == y) / y.size


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)

X_train = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
y_train = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

network = [
    Dense(2, 3),
    Leaky_ReLU(),
    Dense(3, 1),
    Leaky_ReLU()
]

iterations = 10000
learning_rate = 0.1

for i in range(iterations):
    cost = 0

    for x, y in zip(X_train, y_train):
        output = x
        for layer in network:
            output = layer.forward(output)

        cost += mse(y, output)

    gradient = mse_prime(y, output)
    for layer in reversed(network):
        gradient = layer.backward(gradient, learning_rate)

    #I have no idea what this does
    cost /= len(x)
    if (i+1)%1000 == 0:
        print(i+1, cost)
