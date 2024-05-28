from sklearn.datasets import load_digits
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

digits = load_digits()

# The data and target are already NumPy arrays, but let's explicitly convert them
data = np.array(digits.data)
target = np.array(digits.target)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=15)

X_train = X_train.T
X_test = X_test.T
y_train = y_train  # y's are vectors, no need to transpose.
y_test = y_test


def init_params(X, y, hidden_layer_1_nodes=10):
    """
    :param X: np.ndarray n×m
    so that n is the number of features and m is the number of training examples

    :param y: np.ndarray 1×m
    note 1×m isn't different from m×1 in the code, just a vector of all examples

    :return:
    """
    num_outputs = len(np.unique(y))  # will notate `o`, o=10 in digits example
    W1 = np.random.randn(hidden_layer_1_nodes, X.shape[0])
    b1 = np.random.randn(hidden_layer_1_nodes, 1)
    W2 = np.random.randn(num_outputs, hidden_layer_1_nodes)
    b2 = np.random.randn(num_outputs, 1)
    return W1, b1, W2, b2


def Leaky_ReLU(Z, a=0.01):
    return np.maximum(a * Z, Z)


def derivative_Leaky_ReLU(Z, a=0.01):
    result = np.where(Z > 0, 1, np.where(Z < 0, a, 0))
    return result



def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z))


def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = Leaky_ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def one_hot(y):
    """
    Takes in a vector (length m) of the target output for each training example.
    Converts it into a onehot matrix with each row
    """
    categoricals = np.sort(np.unique(y))  # output layer will be in the order of this array
    dimensions = y.size, categoricals.size
    one_hot_y = np.zeros(dimensions)
    one_hot_y[np.arange(categoricals.size), y] = 1
    return one_hot_y.T  # so that it is o×m  where o is the number of possible outputs


def back_prop(Z1, A1, Z2, A2, W2, X, y):
    one_hot_y = one_hot(y)
    m = y.size
    dC_dW2 = A1 * derivative_Leaky_ReLU(Z2) * (2 * (A2 - one_hot_y))
    dC_db2 =A1 * derivative_Leaky_ReLU(Z2) * (2 * (A2 - one_hot_y))
    return dC_dW1, dC_db1, dC_dW2, dC_db2


def update_params(W1, b1, W2, b2, dC_dW1, dC_db1, dC_dW2, dC_db2, alpha):
    W1 -= alpha * dC_dW1
    b1 -= alpha * dC_db1
    W2 -= alpha * dC_dW2
    b2 -= alpha * dC_db2
    return W1, b1, W2, b2


def get_predictions(A2):
    return np.argmax(A2, 0)


def get_accuracy(predictions, y):
    return np.sum(predictions == y) / y.size


def neural_network(X, y, iterations, alpha, W1_nodes):
    W1, b1, W2, b2 = init_params(X, y, hidden_layer_1_nodes=10)
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X, )
        dC_dW1, dC_db1, dC_dW2, dC_db2 = back_prop(Z1, A1, Z2, A2, W2, X, y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dC_dW1, dC_db1, dC_dW2, dC_db2, alpha)
        if i % 100 == 0:
            print("Iteration:", i)
            print("Accuracy:", get_accuracy(get_predictions(A2), y))

init_params(X_train, y_train)
