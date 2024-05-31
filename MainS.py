import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def display_mnist_image(data, image_size=(8, 8), invert_colors=True):
    """
    Converts a flat MNIST data array into an image and displays it using Matplotlib.

    Parameters:
    data (numpy.ndarray): A 1D array of pixel values.
    image_size (tuple): The dimensions of the image (height, width). Default is (8, 8).

    Returns:
    None
    """
    # Check if the data length matches the expected image size
    assert len(data) == image_size[0] * image_size[1], "Data length does not match the expected image size."

    # Reshape the data into a 2D array
    image = data.reshape(image_size)
    if invert_colors:
        image = 1.0 - image

    # Display the image using Matplotlib
    plt.imshow(image, cmap='gray', interpolation='nearest')
    plt.axis('off')  # Hide the axes
    plt.show()


def one_hot(y_vector):
    """
    Takes in a vector (length m) of the target output for each training example.
    Converts it into a one-hot matrix with each row.
    """
    categoricals = np.sort(np.unique(y_vector))  # output layer will be in the order of this array
    dimensions = (y_vector.size, categoricals.size)
    one_hot_y = np.zeros(dimensions)

    # Create a dictionary to map the values in y_vector to the indices in categoricals
    value_to_index = {value: idx for idx, value in enumerate(categoricals)}

    # Populate the one_hot_y matrix
    for row, value in enumerate(y_vector):
        col = value_to_index[value]
        if col != value:
            print('tg')
        one_hot_y[row, col] = 1

    # for yy in one_hot_y.T[]:
    #     print(yy.T)
    # exit()
    return one_hot_y


def normalize_rows(arr, a, b):
    # Compute the minimum and maximum of the

    # Normalize rows to the range [0, 1]
    normalized_rows = (arr - a) / (b - a)

    # Scale and shift to the range [a, b]
    scaled_rows = a + (normalized_rows * (b - a))

    return scaled_rows


def load_data(normalize=True):
    digits = load_digits()
    target = digits.target
    data = digits.data

    global_min = np.min(data)
    data = (data - global_min) / (np.max(data) - global_min)  # normalize
    target = one_hot(target)

    # X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=21)
    X_train, y_train = data, target

    X_train = X_train[:, :, np.newaxis]
    X_test = None  # X_test[:, :, np.newaxis]
    y_train = y_train[:, np.newaxis]
    y_test = None  # y_test[:, np.newaxis]
    return X_train, X_test, y_train, y_test


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input_):
        # TODO: return output
        pass

    def backward(self, output_gradient, learning_rate):
        # TODO: update parameters and return input gradient
        pass


class Dense(Layer):
    def __init__(self, input_size, output_size):
        """output size is the next layer dimension, input size is how much data it is taking in"""
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input_):
        self.input = input_
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


def applyNeuralNetwork(network, X):
    for layer in network:
        X = layer.forward(X)
    return X


def main():
    X_train, X_test, y_train, y_test = load_data()

    network = [
        Dense(64, 20),
        Leaky_ReLU(),
        Dense(20, 10),
        Leaky_ReLU()
    ]

    iterations = 30000
    learning_rate = 0.01

    for i in range(iterations):
        cost = 0

        for x, y in zip(X_train, y_train):
            y = y.T
            output = x
            for layer in network:
                output = layer.forward(output)

            cost += mse(y, output)

        gradient = mse_prime(y, output)
        for layer in reversed(network):
            gradient = layer.backward(gradient, learning_rate)

        cost /= len(x)  # average the cost of all training examples (see above where cost +=...)
        if (i+1)%1000 == 0:
            print(i+1, cost)
    test_input = X_train[5]
    print(f"correct answer: {list(y_train[5][0]).index(1.)}")

    print(applyNeuralNetwork(network, test_input))
    display_mnist_image(test_input)


# test_input = np.reshape([[0, 0]], (2, 1))  # Example input

main()
# X_train, X_test, y_train, y_test = load_data()
# a = 5
# test_input = X_test[a]
# print(y_test[a])

# print(f'Input: {test_input.flatten()} Gives output: {applyNeuralNetwork(network, test_input)}')
