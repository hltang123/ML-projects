from ast import AsyncFunctionDef
from os import O_WRONLY
import sys
import numpy as np
import math

class Node:
    '''
    Here is an arbitrary Node class that will form the basis of your decision
    tree. 
    Note:
        - the attributes provided are not exhaustive: you may add and remove
        attributes as needed, and you may allow the Node to take in initial
        arguments as well
        - you may add any methods to the Node class if desired 
    '''
    def __init__(self, attr, v):
        self.left = None
        self.right = None
        self.leftlabel = None
        self.attr = attr
        self.vote = v

def majority(values):
    labels, counts = np.unique(values, return_counts=True)
    if counts[0] > counts[1]:
        return labels[0]
    if counts[1] > counts[0]:
        return labels[1]
    if labels[0] > labels[1]:
        return labels[0]
    return labels[1]

def entropy(values):
    _, counts = np.unique(values, return_counts=True)
    total_count = np.sum(counts)
    P = counts/total_count
    H = 0
    for p in P:
        H = H - p * math.log2(p)
    return H

def create_DT(train_data, depth):
    num_rows = train_data.shape[0]
    num_cols = train_data.shape[1]
    H_Y = entropy(np.array(train_data[1:,num_cols-1]))
    if H_Y == 0:
        if num_rows == 1:
            return Node(train_data[0,num_cols-1], "lmao")
        return Node(train_data[0,num_cols-1], train_data[1,num_cols-1])
    if depth == 0 or num_cols == 1:
        mvote = majority(np.array(train_data[1:,num_cols-1]))
        return Node(train_data[0, num_cols-1], mvote)
    def minfo(X):
        labels, counts = np.unique(train_data[1:,X], return_counts=True)
        total_count = np.sum(counts)
        H_YX = 0
        for i in range(len(labels)):
            filtered = []
            for x in range(total_count):
                if train_data[x+1,X] == labels[i]:
                    filtered.append(train_data[x+1,num_cols-1])
            H_YX = H_YX + (counts[i] / total_count) * entropy(np.array(filtered))
        return H_Y - H_YX
    I_YX = (minfo(X) for X in range(num_cols-1))
    I = np.fromiter(I_YX, float)
    split_idx = np.argmax(I)
    labels = np.unique(train_data[1:,split_idx])
    DT = Node(train_data[0,split_idx], None)
    DT.leftlabel = labels[0]
    leftfilter = [0]
    rightfilter = [0]
    for x in range(num_rows-1):
        if train_data[x+1,split_idx] == labels[0]:
            leftfilter.append(x+1)
        else:
            rightfilter.append(x+1)
    vfilter = []
    for x in range(num_cols):
        if x != split_idx:
            vfilter.append(x)
    ldata = train_data[leftfilter,:]
    ldata2 = ldata[:,vfilter]
    rdata = train_data[rightfilter,:]
    rdata2 = rdata[:,vfilter]
    DT.left = create_DT(ldata2, depth-1)
    DT.right = create_DT(rdata2, depth-1)
    return DT

def predicter(DT, dict):
    if DT.vote != None:
        return DT.vote
    if dict[DT.attr] == DT.leftlabel:
        return predicter(DT.left, dict)
    return predicter(DT.right, dict)

def predict(DT, data):
    predictions = []
    for i in range(data.shape[0]-1):
        attributes = {}
        for j in range(data.shape[1]):
            attributes[data[0,j]] = data[i+1,j]
        predictions.append(predicter(DT, attributes))
    return predictions

def find_error(predictions, actual):
    total = len(predictions)
    wrong = 0
    for i in range(total):
        if predictions[i] != actual[i]:
            wrong = wrong + 1
    return wrong / total

def pprint(DT):
    print("DT goes here\n")

if __name__ == '__main__':
    pass

# Get all of the arguments from command line.
args = sys.argv

# Parse every argument
train_input = args[1]
test_input = args[2]
max_depth = int(args[3])
train_output = args[4]
test_output = args[5]
metrics_output = args[6]

# Read training and test data
with open(train_input, 'r') as train_in:
    train_data = np.genfromtxt(train_in, dtype = None, delimiter = '\t', encoding = None)
with open(test_input, 'r') as test_in:
    test_data = np.genfromtxt(test_in, dtype = None, delimiter = '\t', encoding = None)

# Create a DT from training data
DT = create_DT(train_data, max_depth)

# Use DT to find predictions and error rates
train_predict = predict(DT, train_data[:,:train_data.shape[1]-1])
with open(train_output, 'w') as train_out:
    for p in train_predict:
        train_out.write(str(p) + "\n")
test_predict = predict(DT, test_data[:,:train_data.shape[1]-1])
with open(test_output, 'w') as test_out:
    for p in test_predict:
        test_out.write(str(p) + "\n")
with open(metrics_output, 'w') as metrics_out:
    metrics_out.write("error(train): " + str(find_error(np.array(train_predict), np.array(train_data[1:,train_data.shape[1]-1]))) + "\n")
    metrics_out.write("error(test): " + str(find_error(np.array(test_predict), np.array(test_data[1:,train_data.shape[1]-1]))) + "\n")

# Pretty print DT
pprint(DT)