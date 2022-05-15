from ast import AsyncFunctionDef
from turtle import update
import numpy as np
import argparse
import logging
import math
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('train_input', type=str,
                    help='path to training input .csv file')
parser.add_argument('validation_input', type=str,
                    help='path to validation input .csv file')
parser.add_argument('train_out', type=str,
                    help='path to store prediction on training data')
parser.add_argument('validation_out', type=str,
                    help='path to store prediction on validation data')
parser.add_argument('metrics_out', type=str,
                    help='path to store training and testing metrics')
parser.add_argument('num_epoch', type=int,
                    help='number of training epochs')
parser.add_argument('hidden_units', type=int,
                    help='number of hidden units')
parser.add_argument('init_flag', type=int, choices=[1, 2],
                    help='weight initialization functions, 1: random')
parser.add_argument('learning_rate', type=float,
                    help='learning rate')
parser.add_argument('--debug', type=bool, default=False,
                    help='set to True to show logging')


def args2data(parser):
    """
    Parse argument, create data and label.
    :return:
    X_tr: train data (numpy array)
    y_tr: train label (numpy array)
    X_te: test data (numpy array)
    y_te: test label (numpy array)
    out_tr: predicted output for train data (file)
    out_te: predicted output for test data (file)
    out_metrics: output for train and test error (file)
    n_epochs: number of train epochs
    n_hid: number of hidden units
    init_flag: weight initialize flag -- 1 means random, 2 means zero
    lr: learning rate
    """

    # # Get data from arguments
    out_tr = parser.train_out
    out_te = parser.validation_out
    out_metrics = parser.metrics_out
    n_epochs = parser.num_epoch
    n_hid = parser.hidden_units
    init_flag = parser.init_flag
    lr = parser.learning_rate

    X_tr = np.loadtxt(parser.train_input, delimiter=',')
    y_tr = X_tr[:, 0].astype(int)
    X_tr[:, 0] = 1.0 #add bias terms

    X_te = np.loadtxt(parser.validation_input, delimiter=',')
    y_te = X_te[:, 0].astype(int)
    X_te[:, 0]= 1.0 #add bias terms


    return (X_tr, y_tr, X_te, y_te, out_tr, out_te, out_metrics,
            n_epochs, n_hid, init_flag, lr)



def shuffle(X, y, epoch):
    """
    Permute the training data for SGD.
    :param X: The original input data in the order of the file.
    :param y: The original labels in the order of the file.
    :param epoch: The epoch number (0-indexed).
    :return: Permuted X and y training data for the epoch.
    """
    np.random.seed(epoch)
    N = len(y)
    ordering = np.random.permutation(N)
    return X[ordering], y[ordering]

def random_init(shape):
    """
    Randomly initialize a numpy array of the specified shape
    :param shape: list or tuple of shapes
    :return: initialized weights
    """
    # DO NOT CHANGE THIS
    np.random.seed(np.prod(shape))

    return np.random.uniform(-0.1, 0.1, shape)


def zero_init(shape):
    """
    Initialize a numpy array of the specified shape with zero
    :param shape: list or tuple of shapes
    :return: initialized weights
    """
    return np.zeros(shape)


class NN(object):
    def __init__(self, lr, n_epoch, weight_init_fn, input_size, hidden_size, output_size):
        """
        Initialization
        :param lr: learning rate
        :param n_epoch: number of training epochs
        :param weight_init_fn: weight initialization function
        :param input_size: number of units in the input layer
        :param hidden_size: number of units in the hidden layer
        :param output_size: number of units in the output layer
        """
        self.lr = lr
        self.n_epoch = n_epoch
        self.weight_init_fn = weight_init_fn
        self.n_input = input_size
        self.n_hidden = hidden_size
        self.n_output = output_size

        # initialize weights and biases for the models
        # HINT: pay attention to bias here
        self.w1 = weight_init_fn([hidden_size, input_size])
        self.w2 = weight_init_fn([output_size, hidden_size+1])

        # initialize parameters for adagrad
        self.epsilon = 0.00001
        self.grad_sum_w1 = np.zeros((hidden_size, input_size))
        self.grad_sum_w2 = np.zeros((output_size, hidden_size+1))

        # feel free to add additional attributes


def print_weights(nn):
    """
    An example of how to use logging to print out debugging infos.

    Note that we use the debug logging level -- if we use a higher logging
    level, we will log things with the default logging configuration,
    causing potential slowdowns.

    Note that we log NumPy matrices on separate lines -- if we do not do this,
    the arrays will be turned into strings even when our logging is set to
    ignore debug, causing potential massive slowdowns.
    :param nn: your model
    :return:
    """
    logging.debug(f"shape of w1: {nn.w1.shape}")
    logging.debug(nn.w1)
    logging.debug(f"shape of w2: {nn.w2.shape}")
    logging.debug(nn.w2)


def sigmoid(x):
    return 1 / (1 + np.exp((-1) * x))

def forward(X, nn):
    """
    Neural network forward computation.
    Follow the pseudocode!
    :param X: input data
    :param nn: neural network class
    :return: output probability
    """
    a = np.matmul(nn.w1, X)
    z = np.vstack((np.matrix(1), sigmoid(a)))
    b = np.matmul(nn.w2, z)
    y_hat = np.exp(b) / np.sum(np.exp(b))
    return y_hat


def backward(X, y, y_hat, nn):
    """
    Neural network backward computation.
    Follow the pseudocode!
    :param X: input data
    :param y: label
    :param y_hat: prediction
    :param nn: neural network class
    :return:
    d_w1: gradients for w1
    d_w2: gradients for w2
    """
    a = np.matmul(nn.w1, X)
    z = np.vstack((np.matrix(1), sigmoid(a)))

    g_b = y_hat - y
    g_beta = np.matmul(g_b, np.transpose(z))
    g_z = np.matmul(np.transpose(nn.w2), g_b)
    g_a = np.multiply(g_z, np.multiply(z, 1 - z))
    g_aa = g_a[1:,:]
    g_alpha = np.matmul(g_aa, np.transpose(X))

    return g_alpha, g_beta


def test(X, y, nn):
    """
    Compute the label and error rate.
    :param X: input data
    :param y: label
    :param nn: neural network class
    :return:
    labels: predicted labels
    error_rate: prediction error rate
    """
    labels = np.copy(y)
    error_num = 0
    for i in range(X.shape[0]):
        y_hat = forward(np.transpose(np.matrix(X[i,:])), nn)
        labels[i] = np.argmax(y_hat)
        if y[i] != labels[i]:
            error_num += 1
    return labels, (error_num / X.shape[0])


def train(X_tr, y_tr, X_val, y_val, nn):
    """
    Train the network using SGD for some epochs.
    :param X_tr: train data
    :param y_tr: train label
    :param nn: neural network class
    """
    xentropy = np.empty((0,2))
    for e in range(nn.n_epoch):
        X_t, labels_t = shuffle(X_tr, y_tr, e)
        X_v, labels_v = shuffle(X_val, y_val, e)

        xent_train = 0
        xent_val = 0
        for i in range(X_t.shape[0]):
            y = np.zeros(10)
            y[labels_t[i]] = 1
            y_hat = forward(np.transpose(np.matrix(X_t[i,:])), nn)
            g_w1, g_w2 = backward(np.transpose(np.matrix(X_t[i,:])), np.transpose(np.matrix(y)), y_hat, nn)
            nn.grad_sum_w1 += np.multiply(g_w1, g_w1)
            nn.grad_sum_w2 += np.multiply(g_w2, g_w2)
            nn.w1 -= nn.lr * np.multiply(1 / np.sqrt(nn.grad_sum_w1 + nn.epsilon), g_w1)
            nn.w2 -= nn.lr * np.multiply(1 / np.sqrt(nn.grad_sum_w2 + nn.epsilon), g_w2)

        for i in range(X_t.shape[0]):
            y_hat = forward(np.transpose(np.matrix(X_t[i,:])), nn)
            xent_train += math.log(y_hat[labels_t[i]])
        for i in range(X_v.shape[0]):
            y_hat = forward(np.transpose(np.matrix(X_v[i,:])), nn)
            xent_val += math.log(y_hat[labels_v[i]])
        
        print("EPOCH " + str(e+1) + ":\n")
        print(nn.w1)
        print(nn.w2)
        
        xentropy = np.vstack((xentropy, np.matrix([(-1) * xent_train / X_t.shape[0], (-1) * xent_val / X_v.shape[0]])))
    return xentropy


# def train2(X_tr, y_tr, X_val, y_val, lr, n_epochs, n_hid):
#     """
#     Train the network using SGD for some epochs.
#     :param X_tr: train data
#     :param y_tr: train label
#     :param nn: neural network class
#     """
#     nn = NN(lr, n_epochs, random_init, X_tr.shape[1], n_hid, 10)
#     for e in range(nn.n_epoch):
#         X_t, labels_t = shuffle(X_tr, y_tr, e)

#         for i in range(X_t.shape[0]):
#             y = np.zeros(10)
#             y[labels_t[i]] = 1
#             y_hat = forward(np.transpose(np.matrix(X_t[i,:])), nn)
#             g_w1, g_w2 = backward(np.transpose(np.matrix(X_t[i,:])), np.transpose(np.matrix(y)), y_hat, nn)
#             nn.grad_sum_w1 += np.multiply(g_w1, g_w1)
#             nn.grad_sum_w2 += np.multiply(g_w2, g_w2)
#             nn.w1 -= nn.lr * np.multiply(1 / np.sqrt(nn.grad_sum_w1 + nn.epsilon), g_w1)
#             nn.w2 -= nn.lr * np.multiply(1 / np.sqrt(nn.grad_sum_w2 + nn.epsilon), g_w2)

#     xent_train = 0
#     xent_val = 0
#     for i in range(X_tr.shape[0]):
#         y_hat = forward(np.transpose(np.matrix(X_tr[i,:])), nn)
#         xent_train += math.log(y_hat[y_tr[i]])
#     for i in range(X_val.shape[0]):
#         y_hat = forward(np.transpose(np.matrix(X_val[i,:])), nn)
#         xent_val += math.log(y_hat[y_val[i]])
        
#     return (-1) * xent_train / X_tr.shape[0], (-1) * xent_val / X_val.shape[0]


# def plot1(X_tr, y_tr, X_te, y_te):
#     tr1, val1 = train2(X_tr, y_tr, X_te, y_te, 0.01, 100, 5)
#     tr2, val2 = train2(X_tr, y_tr, X_te, y_te, 0.01, 100, 20)
#     tr3, val3 = train2(X_tr, y_tr, X_te, y_te, 0.01, 100, 50)
#     tr4, val4 = train2(X_tr, y_tr, X_te, y_te, 0.01, 100, 100)
#     tr5, val5 = train2(X_tr, y_tr, X_te, y_te, 0.01, 100, 200)

#     tr = [tr1, tr2, tr3, tr4, tr5]
#     val = [val1, val2, val3, val4, val5]

#     plt.title("Average Cross-Entropy vs Number of Hidden Units")
#     plt.xlabel("Hidden Units")
#     plt.ylabel("Average Cross-Entropy")
#     x = [5, 20, 50, 100, 200]
#     xi = list(range(len(x)))
#     plt.plot(x, tr, "r", linewidth=3.0, label="average cross-entropy (train)")
#     plt.plot(x, val, linewidth=3.0, label="average cross-entropy (validation)")
#     # plt.xticks(xi, x)
#     plt.legend(loc="upper right")
#     plt.show()
    
# def plot2(X_tr, y_tr, X_te, y_te):
#     nn = NN(0.01, 100, random_init, X_tr.shape[1], 50, 10)
#     xent = train(X_tr, y_tr, X_te, y_te, nn)

#     val1 = [2.2880272073923806,2.238260441710385,2.1765565439591628,2.0468370179140534,1.9162777073144173,1.7887433596941775,1.7030114470162832,1.6105032408887334,1.5158290108113868,1.436833524454427,1.3744921873714504,1.3215078576000692,1.2643357354715667,1.2017988996385247,1.168830429444711,1.1354975144450958,1.1164518726302617,1.10489073386534,1.087918871026834,1.0509291453511214,1.058311312430424,1.0622188456512,1.0348516415991549,1.0366041556473444,1.0314503714882413,1.0514406238875245,1.0298936469069557,1.0303446021448912,1.0441762302962776,1.0321361912976008,1.0430247356054243,1.0446465529779319,1.043938059176376,1.055320277419167,1.0474271146234413,1.0459968045202634,1.0627312352004095,1.0639804208713164,1.0727641436050086,1.0687702538632167,1.084654502353079,1.08344335170731,1.0842655056498112,1.1008342610867436,1.0898234570889165,1.105480844381485,1.1488110684800967,1.133093405656082,1.123589500288112,1.1381090820654747,1.1392865588961005,1.1520027547228289,1.1563087281003683,1.1696636345478746,1.1818773374607918,1.1853498195563332,1.1725295779380054,1.2043063631234137,1.1902218687192405,1.1987795507145433,1.2088191587629993,1.2121158621030605,1.2114426537345035,1.211490864688144,1.225384431384039,1.2296892282863334,1.233217452023371,1.2431171178020728,1.2431013528623809,1.2487490677039954,1.267691846325217,1.2787245110897547,1.2605601145764325,1.2796890537986612,1.279031470327852,1.2946269180740684,1.2947446716080844,1.2977926048924209,1.293367883773991,1.3121508522700103,1.3227590972316796,1.3122094418451775,1.3276651168727522,1.3247338047568455,1.3265507365592217,1.3337437942394195,1.3567910257328932,1.3448127175699258,1.3475524830839305,1.361918180587943,1.371782378195801,1.3541701017884575,1.3575264290438636,1.3767213222199535,1.371035544364968,1.3800399338480422,1.3936676816727567,1.3918085562816902,1.4063750237113561,1.3887846857071842]
#     val2 = np.array(xent[:,1])

#     plt.title("Average Cross-Entropy vs Epoch")
#     plt.xlabel("Epoch")
#     plt.ylabel("Average Cross-Entropy")
#     x = np.arange(1, 101)
#     plt.plot(x, val1, "r", linewidth=3.0, label="average cross-entropy (without Adagrad)")
#     plt.plot(x, val2, linewidth=3.0, label="average cross-entropy (with Adagrad)")
#     plt.legend(loc="upper right")
#     plt.show()
    
# def plot3(X_tr, y_tr, X_te, y_te):
#     nn = NN(0.01, 100, random_init, X_tr.shape[1], 50, 10)
#     xent = train(X_tr, y_tr, X_te, y_te, nn)

#     tr = np.array(xent[:,0])
#     val = np.array(xent[:,1])

#     plt.title("Average Cross-Entropy vs Epoch (learning rate = 0.01)")
#     plt.xlabel("Epoch")
#     plt.ylabel("Average Cross-Entropy")
#     x = np.arange(1, 101)
#     plt.plot(x, tr, "r", linewidth=3.0, label="average cross-entropy (train)")
#     plt.plot(x, val, linewidth=3.0, label="average cross-entropy (validation)")
#     plt.legend(loc="upper right")
#     plt.show()

if __name__ == "__main__":

    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(format='[%(asctime)s] {%(pathname)s:%(funcName)s:%(lineno)04d} %(levelname)s - %(message)s',
                            datefmt="%H:%M:%S",
                            level=logging.DEBUG)
    logging.debug('*** Debugging Mode ***')
    # Note: You can access arguments like learning rate with args.learning_rate

    # initialize training / test data and labels
    (X_tr, y_tr, X_te, y_te, out_tr, out_te, out_metrics, n_epochs, n_hid, init_flag, lr) = args2data(args)

    # plot3(X_tr, y_tr, X_te, y_te)

    # Build model
    if init_flag == 1:
        init_fn = random_init
    else:
        init_fn = zero_init
    my_nn = NN(lr, n_epochs, init_fn, X_tr.shape[1], n_hid, 10)

    # train model
    xent = train(X_tr, y_tr, X_te, y_te, my_nn)
    train_labels, train_error_rate = test(X_tr, y_tr, my_nn)

    # test model and get predicted labels and errors
    val_labels, val_error_rate = test(X_te, y_te, my_nn)

    # write predicted label and error into file
    with open(out_tr, 'w') as fout:
        for i in range(train_labels.shape[0]):
            fout.write(str(train_labels[i]) + "\n")
    with open(out_te, 'w') as fout:
        for i in range(val_labels.shape[0]):
            fout.write(str(val_labels[i]) + "\n")
    with open(out_metrics, 'w') as fout:
        for i in range(n_epochs):
            fout.write("epoch=" + str(i+1) + " crossentropy(train): " + str(xent[i,0]) + "\n")
            fout.write("epoch=" + str(i+1) + " crossentropy(validation): " + str(xent[i,1]) + "\n")
        fout.write("error(train): " + "{:.6f}".format(train_error_rate) + "\n")
        fout.write("error(test): " + "{:.6f}".format(val_error_rate) + "\n")
