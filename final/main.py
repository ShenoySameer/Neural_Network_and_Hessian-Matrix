import numpy as np
from sklearn.datasets import load_digits

from final.backend import mse, mse_prime, Leaky_ReLU, Softmax
from helperFunctions import display_mnist_image, build_network, one_hot

digits = load_digits()
X = digits.data
y = digits.target

iterations = 10000
learning_rate = 0.1
network = build_network(X, y, Leaky_ReLU(.01),
                        hidden_layers_neurons=[16, 16, 16],)
                        # final_activ=Softmax())
Y = one_hot(y)

for iteration in range(iterations):
    cost = 0
    for i in range(len(y)):
        Xi = X[i][:, np.newaxis]
        Yi = Y[i][:, np.newaxis]
        output = Xi
        for layer in network:
            # print(output.shape)
            output = layer.forward(output)

        cost += mse(Yi, output)  # error

        # backward propagation:
        gradient = mse_prime(Yi, output)

        for layer in network[::-1]:
            # print(layer)
            # print(gradient.shape)
            gradient = layer.backward(gradient, learning_rate)

    cost /= len(y)

    if iteration % 1000:
        print(f'Iteration: {iteration}/{iterations}\tError: {cost}')
