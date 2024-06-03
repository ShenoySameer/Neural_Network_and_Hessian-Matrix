import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

digits = load_digits()
data = np.array(digits.data)
target = np.array(digits.target)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=15)

X_train = X_train.T
X_test = X_test.T

# Normalize the input data
X_train = X_train / 16.0
X_test = X_test / 16.0

y_train = y_train  # y's are vectors, no need to transpose.
y_test = y_test

X_train = np.reshape(X_train, (64, 1437, 1)).T

class Dense():
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2. / input_size)  # He initialization
        self.bias = np.zeros((output_size, 1))

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient

class Activation():
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
        self.input = None
        self.output = None

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))

class Leaky_ReLU(Activation):
    def __init__(self, a=0.01):
        self.a = a
        lr = lambda x: np.maximum(self.a * x, x)
        lr_prime = lambda x: np.where(x > 0, 1, self.a)
        super().__init__(lr, lr_prime)

def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

network = [
    Dense(64, 16),
    Leaky_ReLU(),
    Dense(16, 10),
    Leaky_ReLU()
]

iterations = 10
learning_rate = 0.001  # Reduced learning rate

for i in range(iterations):
    cost = 0

    for x, y in zip(X_train, y_train):
        print(x.shape)
        output = x
        for layer in network:
            output = layer.forward(output)

        cost += mse(y, output)
        gradient = mse_prime(y, output)
        for layer in reversed(network):
            gradient = layer.backward(gradient, learning_rate)

    cost /= len(X_train)
    if (i+1) % 10 == 0:
        print(f"Iteration {i+1}, Cost: {cost}")
        if np.isnan(cost):
            print("NaN encountered!")
            break

def applyNeuralNetwork(network, X):
    for layer in network:
        X = layer.forward(X)
    return X

test_input = X_test[:, :, np.newaxis][0]
print(f'Input: {test_input.flatten()} Gives output: {applyNeuralNetwork(network, X_test)}')
print(y_test[0])
