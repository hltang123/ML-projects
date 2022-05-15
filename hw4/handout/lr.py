from operator import truediv
import sys
import numpy as np


def sigmoid(x):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (str): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(x)
    return e / (1 + e)


def train(theta, X, y, num_epoch, learning_rate):
    for epoch in range(num_epoch):
        for i in range(X.shape[0]):
            theta = theta + learning_rate * X[i] * (y[i] - sigmoid(np.dot(theta, X[i])))
    return theta


def predict(theta, X):
    def prediction(theta, x):
        prod = (-1) * np.dot(theta, x)
        if sigmoid(prod) >= 0.5:
            return 0
        return 1
    iterable = (prediction(theta, X[i]) for i in range(X.shape[0]))
    return np.fromiter(iterable, int)


def compute_error(y_pred, y):
    diff = 0
    for i in range(y_pred.shape[0]):
        if y_pred[i] != y[i]:
            diff = diff + 1
    return diff / y_pred.shape[0]

if __name__ == '__main__':
    pass

# Get all of the arguments from command line.
args = sys.argv

# Parse every argument
formatted_train_input = args[1]
formatted_validation_input = args[2]
formatted_test_input = args[3]
train_out = args[4]
test_out = args[5]
metrics_out = args[6]
num_epoch = int(args[7])
learning_rate = float(args[8])

# Read training, validation, and test data
with open(formatted_train_input, 'r') as train_in:
    train_data = np.genfromtxt(train_in, dtype = None, delimiter = '\t', encoding = None)
# with open(formatted_validation_input, 'r') as valid_in:
#     validation_data = np.genfromtxt(valid_in, dtype = None, delimiter = '\t', encoding = None)
with open(formatted_test_input, 'r') as test_in:
    test_data = np.genfromtxt(test_in, dtype = None, delimiter = '\t', encoding = None)

# Parse data
X_train = np.copy(train_data)
X_train[:, 0] = 1
y_train = train_data[:, 0]
# X_validation = np.copy(validation_data)
# X_validation[:, 0] = 1
# y_validation = validation_data[:, 0]
X_test = np.copy(test_data)
X_test[:, 0] = 1
y_test = test_data[:, 0]
theta_init = np.zeros(X_train.shape[1])

# Train using linear regression
theta = train(theta_init, X_train, y_train, num_epoch, learning_rate)

# Predict on the validation and test data sets and report results
train_pred = predict(theta, X_train)
with open(train_out, 'w') as fout:
    for i in range(train_pred.shape[0]):
        fout.write(str(train_pred[i]) + "\n")

test_pred = predict(theta, X_test)
with open(test_out, 'w') as fout:
    for i in range(test_pred.shape[0]):
        fout.write(str(test_pred[i]) + "\n")

train_error = compute_error(train_pred, y_train)
test_error = compute_error(test_pred, y_test)
with open(metrics_out, 'w') as fout:
    fout.write("error(train): " + "{:.6f}".format(train_error) + "\n")
    fout.write("error(test): " + "{:.6f}".format(test_error) + "\n")