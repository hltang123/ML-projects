import sys
import numpy as np

# Get all of the arguments from command line and make sure 
# the number of arguments is correct.
args = sys.argv
assert(len(args) == 6)

# Parse every argument
train_input = args[1]
test_input = args[2]
train_output = args[3]
test_output = args[4]
metrics_output = args[5]

# Open training data and find majority vote
majority = ""
with open(train_input, 'r') as file:
    data = np.genfromtxt(file, delimiter="\t", dtype=None, encoding=None)
    len = data.shape[1]
    votes = data[1:,len-1:len]
    a = np.array(votes)
    values, counts = np.unique(a, return_counts=True)
    mvote_val = np.max(counts[:])
    temp = np.argwhere(counts == mvote_val)
    mvote_idx = temp.flatten()
    mvotes = values[mvote_idx]
    sorted = np.sort(mvotes)
    majority = sorted[sorted.shape[0] - 1]

# Run majority vote algorithm on training data
train_total = 0
train_error = 0
with open(train_input, 'r') as fin:
    data = np.genfromtxt(fin, delimiter="\t", dtype=None, encoding=None)
    len = data.shape[1]
    train_total = data.shape[0] - 1
    with open(train_output, 'w') as fout:
        for x in range(train_total):
            fout.write(majority + "\n")
            if (data[x + 1, len - 1] != majority):
                train_error += 1

# Run majority vote algorithm on test data
test_total = 0
test_error = 0
with open(test_input, 'r') as fin:
    data = np.genfromtxt(fin, delimiter="\t", dtype=None, encoding=None)
    len = data.shape[1]
    test_total = data.shape[0] - 1
    with open(test_output, 'w') as fout:
        for x in range(test_total):
            fout.write(majority + "\n")
            if (data[x + 1, len - 1] != majority):
                test_error += 1

# Write error data into metrics file
with open(metrics_output, 'w') as f:
    f.write("error(train): " + str(train_error/train_total) + "\n")
    f.write("error(test): " + str(test_error/test_total) + "\n")
