import numpy as np
import pandas as pd
from scipy.stats import norm

np.random.seed(1)
trainData = pd.read_csv("./bank-note/train.csv", header=None)
testData = pd.read_csv("./bank-note/test.csv", header=None)

X_train = trainData.iloc[:, :-1].to_numpy()
y_train = trainData.iloc[:, -1].to_numpy()
X_test = testData.iloc[:, :-1].to_numpy()
y_test = testData.iloc[:, -1].to_numpy()

X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

# w0 = np.zeros(5)
# w1 = np.ones(5)
w0 = np.random.normal(loc=0, scale=1, size=5)
w1 = np.random.normal(loc=0, scale=1, size=5)

tolerance = 1e-5
max_iteration = 1000
i=0

while np.linalg.norm(w1-w0) >= tolerance:
    if i <= max_iteration:
        y = 1 / (1 + np.exp(-np.dot(X_train, w0)))
        W = np.diag(y * (1 - y))
        H = np.dot(X_train.T, np.dot(W, X_train))
        # Add Hessian of the prior term
        H += np.identity(X_train.shape[1])
        g = np.dot(X_train.T, y - y_train)
        g += w0
        w_new = w0 - np.dot(np.linalg.inv(H), g)
        w0 = w1
        w1 = w_new
        i += 1
    else:
        break

threshold = 0.5
y_pred = (1 / (1 + np.exp(-np.dot(X_test, w1)))) >= threshold

accuracy = np.sum(y_pred == y_test) / y_test.shape[0]

print("Accuracy on test data with initial weights set to zero: {:.2f}%".format(accuracy * 100))
