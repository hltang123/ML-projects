import sys
import numpy as np
import math

# Get all of the arguments from command line.
args = sys.argv

def find_entropy(counts):
    total_count = np.sum(counts)
    P = counts/total_count
    entropy = 0
    for p in P:
        entropy = entropy - p * math.log2(p)
    return entropy

def find_error(counts):
    total_count = np.sum(counts)
    P = counts/total_count
    P_mvote = np.max(P[:])
    error = 1 - P_mvote
    return error

# Parse every argument
train_input = args[1]
inspect_output = args[2]

# Open training data
with open(train_input, 'r') as fin:
    data = np.genfromtxt(fin, dtype = None, delimiter = '\t', encoding = None)
len = data.shape[1]
votes = data[1:,len-1:len]
a = np.array(votes)
values, counts = np.unique(a, return_counts=True)
entropy = find_entropy(counts)
error = find_error(counts)
with open(inspect_output, 'w') as fout:
    fout.write("entropy: " + str(entropy) + "\n")
    fout.write("error: " + str(error) + "\n")