import numpy as np
from matplotlib import pyplot as plt

from final.backend import Activation, Dense


def build_network(X: np.ndarray, y: np.ndarray, activation: Activation, hidden_layers_neurons: list, final_activ=None):
    """
    hidden_layers_neurons example:
    hidden_layers_neurons = [2, 3, 4]
    means that there will be 3 hidden layers (`len(hidden_layers_neurons) = 3`)
    so that
        the input layer connects to
        a hidden layer with 2 neurons

        which, in turn, connects to
        a hidden layer with 3 neurons

        which, in turn, connects to
        a hidden layer with 4 neurons

        which, in turn, connects to
        the output layer


    This assumes that X is m×n where
    m is the number of training examples and n is the number of features
    """
    if final_activ is None:
        final_activ = activation
    len_input = X.shape[1]
    len_output = len(np.unique(y))

    layer_sizes = [len_input]
    for hidden_layer_neurons in hidden_layers_neurons:
        layer_sizes.append(hidden_layer_neurons)
        layer_sizes.append(hidden_layer_neurons)
    layer_sizes.append(len_output)

    network = []
    for i in range(len(hidden_layers_neurons)):
        layer_size = layer_sizes[2 * i], layer_sizes[2 * i + 1]
        network.append(Dense(*layer_size))
        network.append(activation)
    network.append(Dense(*layer_sizes[-2:]))
    network.append(final_activ)
    # print(network)
    # exit()
    return network


def apply_network(network, X):
    for layer in network:
        X = layer.forward(X)
    return X


def get_predictions(A2):
    return np.argmax(A2, 0)


def get_accuracy(predictions, y):
    return np.sum(predictions == y) / y.size


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
        one_hot_y[row, col] = 1
    return one_hot_y