import sys
import csv
import numpy as np

VECTOR_LEN = 300   # Length of word2vec vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and word2vec.txt

################################################################################
# We have provided you the functions for loading the tsv and txt files. Feel   #
# free to use them! No need to change them at all.                             #
################################################################################


def load_tsv_dataset(file):
    """
    Loads raw data and returns a tuple containing the reviews and their ratings.

    Parameters:
        file (str): File path to the dataset tsv file.

    Returns:
        An N x 2 np.ndarray. N is the number of data points in the tsv file. The
        first column contains the label integer (0 or 1), and the second column
        contains the movie review string.
    """
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8',
                         dtype='l,O')
    return dataset


def load_dictionary(file):
    """
    Creates a python dict from the model 1 dictionary.

    Parameters:
        file (str): File path to the dictionary for model 1.

    Returns:
        A dictionary indexed by strings, returning an integer index.
    """
    dict_map = np.loadtxt(file, comments=None, encoding='utf-8',
                          dtype=f'U{MAX_WORD_LEN},l')
    return {word: index for word, index in dict_map}


def load_feature_dictionary(file):
    """
    Creates a map of words to vectors using the file that has the word2vec
    embeddings.

    Parameters:
        file (str): File path to the word2vec embedding file for model 2.

    Returns:
        A dictionary indexed by words, returning the corresponding word2vec
        embedding np.ndarray.
    """
    word2vec_map = dict()
    with open(file) as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            word, embedding = row[0], row[1:]
            word2vec_map[word] = np.array(embedding, dtype=float)
    return word2vec_map

def bag_of_words(data, dict, outfile):
    with open(outfile, 'w') as fout:
        for i in range(data.shape[0]):
            features = np.zeros(len(dict), dtype=int)
            review = data[i][1].split()
            for word in review:
                if word in dict and features[dict[word]] == 0:
                    features[dict[word]] = 1
            fout.write(str(data[i][0]))
            for attr in features:
                fout.write("\t" + str(attr))
            fout.write("\n")

def word_embeddings(data, dict, outfile):
    with open(outfile, 'w') as fout:
        for i in range(data.shape[0]):
            features = np.zeros(300)
            review = data[i][1].split()
            wordcount = 0
            for word in review:
                if word in dict:
                    features += dict[word]
                    wordcount += 1
            features /= wordcount
            fout.write("{:.6f}".format(data[i][0]))
            for attr in features:
                fout.write("\t" + "{:.6f}".format(attr))
            fout.write("\n")

if __name__ == '__main__':
    pass

# Get all of the arguments from command line.
args = sys.argv

# Parse every argument
train_input = args[1]
validation_input = args[2]
test_input = args[3]
dict_input = args[4]
feature_dictionary_input = args[5]
formatted_train_out = args[6]
formatted_validation_out = args[7]
formatted_test_out = args[8]
feature_flag = int(args[9])

# Read training, validation, and test data
train_data = load_tsv_dataset(train_input)
validation_data = load_tsv_dataset(validation_input)
test_data = load_tsv_dataset(test_input)

# Extract features on all data sets based on feature flag
if feature_flag == 1:
    dict_data = load_dictionary(dict_input)
    bag_of_words(train_data, dict_data, formatted_train_out)
    bag_of_words(validation_data, dict_data, formatted_validation_out)
    bag_of_words(test_data, dict_data, formatted_test_out)

if feature_flag == 2:
    feat_data = load_feature_dictionary(feature_dictionary_input)
    word_embeddings(train_data, feat_data, formatted_train_out)
    word_embeddings(validation_data, feat_data, formatted_validation_out)
    word_embeddings(test_data, feat_data, formatted_test_out)