import numpy as np


class Layer():
    def __init__(self):
        self.input_ = None
        self.output = None

    def forward(self, input_):
        """this is just to be passed down to subclasses"""
        pass

    def backward(self, output_gradient, learning_rate):
        """this is just to be passed down to subclasses"""
        pass


class Dense(Layer):
    def __init__(self, input_size, output_size):
        """output size is the next layer dimension, input size is how much data it is taking in"""
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input_):
        """
        input should be a vector of:
        In the first iteration, all features
        In further iterations, all previous neurons

        output should be a vector of all neurons in the next layer.
        For neuron `k` where the previous layer was `input_size` neurons, this should be:
        b_k + (sum from 1 to input_size) (w__k_n)
        b_k and output will be `(out)×1 vectors`, W will be `(out)×(in)`, in will be `(in)×1`
        thus in linear algebra, this will be `out` = `W dot in` + `b`
        """
        self.input_ = input_  # for later
        # print(self.weights.shape, input_.shape, '--->', (np.dot(self.weights, input_) + self.bias).shape)
        # print(self.weights)
        # print(self.input_)
        # print(np.dot(self.weights, input_) + self.bias)
        # exit()
        return np.dot(self.weights, input_) + self.bias

    def backward(self, dC_dY, learning_rate):
        """
        inputs a vector (called `dC_dY`), outputs the gradient vector of the input (`dC_dX`).
        first output_gradient will be mse_prime(output, y)
        """
        # print(dC_dY.shape)
        X_transpose = self.input_.T
        dC_dW = np.dot(dC_dY, X_transpose)
        dC_dB = dC_dY  # unnecessary assignment, but can be shown with some calculus

        self.weights -= dC_dW * learning_rate  # these matrices have the same dimensions
        self.bias -= dC_dB * learning_rate     # ^

        W_transpose = self.weights.T
        dC_dX = np.dot(W_transpose, dC_dY)
        print("dC_dX.shape", dC_dX.shape)
        return dC_dX

    def __repr__(self):
        return f"Dense({self.input_size}, {self.output_size})"

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def __call__(self, x):
        return self.activation(x)

    def __repr__(self):
        return "Activation())"

    def forward(self, input_):
        self.input_ = input_
        return self.activation(self.input_)

    def backward(self, dC_dY, learning_rate):
        f_prime_of_X = self.activation_prime(self.input_)
        print("ACTIVATION", dC_dY.shape)
        # print(dC_dY)
        dC_dX = np.multiply(dC_dY, f_prime_of_X)
        return dC_dX


class Leaky_ReLU(Activation):
    def __init__(self, a=0):
        """
        :param a: only hyperparameter for this layer, should always be 0≤a<1
        """
        self.a = a
        def f(x):
            """
            see why: https://desmos.com/calculator/w6uzspct2a
            """
            return np.maximum(x * a, x)

        def f_prime(x):
            """
            when x < 0, (np.sign(x) = -1) < a (which is always positive), thus chooses a
            when x > 0, (np.sign(x) = 1) > a (which should always be 0≤a<1), thus chooses 1
            a and 1 are the respective slopes of the function before and after x=0
            """
            return np.maximum(a, np.sign(x))

        super().__init__(f, f_prime)

    def __repr__(self):
        return f"LeakyReLU(a={self.a})"


class Softmax():
    def __init__(self):
        def f(x):
            # ChatGPT:
            exp_values = np.exp(x - np.max(x, axis=0, keepdims=True))
            return exp_values / np.sum(exp_values, axis=0, keepdims=True)
            # Isaac:
            # numerator = np.exp(x)  # array
            # denominator = np.sum(numerator)  # constant
            # return numerator / denominator  # array / constant  is just dividing each item by the sum of the array

        def f_prime(x):
            # ChatGPT:
            softmax_output = f(x)
            return softmax_output * (1 - softmax_output)  # Derivative of softmax is softmax * (1 - softmax)
            # Isaac:
            # return np.sum(np.exp(x))  # just denominator (see `f(x)`)

        super().__init__(f, f_prime)


def mse(y_true, y_pred):
    squared_error = np.square(y_pred - y_true)
    return np.mean(squared_error)


def mse_prime(y_true, y_pred):
    """
    Derivative of mse with respect to y_pred
    """
    derivative_of_squared_error = 2 * (y_pred - y_true)
    # taking the mean is multiplying by 1/(length of y vector), which is a constant, so:
    return (derivative_of_squared_error) / len(y_pred)
