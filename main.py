from sklearn.datasets import load_digits
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

digits = load_digits()

# The data and target are already NumPy arrays, but let's explicitly convert them
data = np.array(digits.data)
target = np.array(digits.target)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=15)

print(X_train.shape)
def init_params():
    w1 = np.random.randn(64, 1437)
    b1
    w2
    b2
