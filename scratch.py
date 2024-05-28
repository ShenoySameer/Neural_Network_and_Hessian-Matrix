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
print(len(y_train))
print(y_train.size)

