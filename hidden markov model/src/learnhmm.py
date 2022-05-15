############ Welcome to HW7 ############
# Andrew-id: hltang


# Imports
# Don't import any other library
import argparse
import numpy as np
from utils import make_dict, parse_file
import logging

# Setting up the argument parser
# don't change anything here

parser = argparse.ArgumentParser()
parser.add_argument('train_input', type=str,
                    help='path to training input .txt file')
parser.add_argument('index_to_word', type=str,
                    help='path to index_to_word.txt file')
parser.add_argument('index_to_tag', type=str,
                    help='path to index_to_tag.txt file')
parser.add_argument('init', type=str,
                    help='path to store the hmm_init.txt (pi) file')
parser.add_argument('emission', type=str,
                    help='path to store the hmm_emission.txt (A) file')
parser.add_argument('transition', type=str,
                    help='path to store the hmm_transition.txt (B) file')
parser.add_argument('--debug', type=bool, default=False,
                    help='set to True to show logging')


# Hint: You might find it useful to define functions that do the following:
# 1. Calculate the init matrix
# 2. Calculate the emission matrix
# 3. Calculate the transition matrix
# 4. Normalize the matrices appropriately

def calc_init(tags, tag_dict):
    pi = np.zeros(len(tag_dict))
    for t in tags:
        pi[tag_dict[t[0]]] += 1
    pi += 1
    pi /= len(tags) + len(tag_dict)
    return pi

def calc_emit(sentences, tags, word_dict, tag_dict):
    A = np.zeros((len(tag_dict), len(word_dict)))
    for i in range(len(sentences)):
        for j in range(len(sentences[i])):
            A[tag_dict[tags[i][j]], word_dict[sentences[i][j]]] += 1
    A += 1
    sums = np.sum(A, axis=1)
    for i in range(A.shape[0]):
        A[i, :] /= sums[i]
    return A

def calc_trans(tags, tag_dict):
    B = np.zeros((len(tag_dict), len(tag_dict)))
    for t in tags:
        for i in range(len(t)-1):
            B[tag_dict[t[i]], tag_dict[t[i+1]]] += 1
    B += 1
    sums = np.sum(B, axis=1)
    for i in range(B.shape[0]):
        B[i, :] /= sums[i]
    return B

# TODO: Complete the main function
def main(args):

    # Get the dictionaries
    word_dict = make_dict(args.index_to_word)
    tag_dict = make_dict(args.index_to_tag)

    # Parse the train file
    # Suggestion: Take a minute to look at the training file,
    # it always hels to know your data :)
    sentences, tags = parse_file(args.train_input)

    logging.debug(f"Num Sentences: {len(sentences)}")
    logging.debug(f"Num Tags: {len(tags)}")
    
    
    # Train your HMM
    N = 10000
    init = calc_init(tags[:N], tag_dict)
    emission = calc_emit(sentences[:N], tags[:N], word_dict, tag_dict)
    transition = calc_trans(tags[:N], tag_dict)

    # Making sure we have the right shapes
    # logging.debug(f"init matrix shape: {init.shape}")
    # logging.debug(f"emission matrix shape: {emission.shape}")
    # logging.debug(f"transition matrix shape: {transition.shape}")


    ## Saving the files for inference
    ## We're doing this for you :)
    ## TODO: Just Uncomment the following lines when you're ready!
    
    np.savetxt(args.init, init)
    np.savetxt(args.emission, emission)
    np.savetxt(args.transition, transition)

    return 

# No need to change anything beyond this point
if __name__ == "__main__":
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(format='[%(asctime)s] {%(pathname)s:%(funcName)s:%(lineno)04d} %(levelname)s - %(message)s',
                            datefmt="%H:%M:%S",
                            level=logging.DEBUG)
    logging.debug('*** Debugging Mode ***')
    main(args)