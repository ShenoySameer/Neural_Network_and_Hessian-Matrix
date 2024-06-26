import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the dataset
digits = load_digits()
data = np.array(digits.data)
target = np.array(digits.target)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=15)

# Normalize the input data
X_train = X_train / 16.0
X_test = X_test / 16.0

# No need to transpose y values
y_train = y_train
y_test = y_test

# Print the shape of the training data
print(X_train.shape)  # Should be (1437, 64)

# Reshape the training data correctly
X_train = X_train.T
print(X_train.shape)  # Should be (64, 1437)

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

iterations = 1000
learning_rate = 0.001  # Reduced learning rate

for i in range(iterations):
    cost = 0

    for x, y in zip(X_train.T, y_train):
        x = x.reshape(64, 1)  # Ensure each input is (64, 1)
        output = x
        for layer in network:
            output = layer.forward(output)

        y = np.eye(10)[y].reshape(10, 1)  # Convert y to one-hot encoding
        cost += mse(y, output)
        gradient = mse_prime(y, output)
        for layer in reversed(network):
            gradient = layer.backward(gradient, learning_rate)

    cost /= len(X_train.T)
    if (i+1) % 10 == 0:
        print(f"Iteration {i+1}, Cost: {cost}")
        if np.isnan(cost):
            print("NaN encountered!")
            break

def applyNeuralNetwork(network, X):
    for layer in network:
        X = layer.forward(X)
    return X

X_test = X_test[:, :, np.newaxis]
total_count = 0
correct_count = 0
for i in range(len(X_test)):
    total_count += 1
    test_input = X_test[i]
    out = applyNeuralNetwork(network, test_input)
    if list(out).index(max(out)) == y_test[i]:
        correct_count += 1
        print('correct')
    else:
        print('incorrect')
print(correct_count/total_count)
# print(f'Input: {test_input.flatten()} Gives output: {applyNeuralNetwork(network, test_input)}')
# print(y_test)