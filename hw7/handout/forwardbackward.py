############ Welcome to HW7 ############
# Andrew-id: hltang


# Imports
# Don't import any other library
from operator import truediv
import numpy as np
from utils import make_dict, parse_file, get_matrices, write_predictions, write_metrics
import argparse
import logging

# Setting up the argument parser
# don't change anything here
parser = argparse.ArgumentParser()
parser.add_argument('validation_input', type=str,
                    help='path to validation input .txt file')
parser.add_argument('index_to_word', type=str,
                    help='path to index_to_word.txt file')
parser.add_argument('index_to_tag', type=str,
                    help='path to index_to_tag.txt file')
parser.add_argument('init', type=str,
                    help='path to the learned hmm_init.txt (pi) file')
parser.add_argument('emission', type=str,
                    help='path to the learned hmm_emission.txt (A) file')
parser.add_argument('transition', type=str,
                    help='path to the learned hmm_transition.txt (B) file')
parser.add_argument('prediction_file', type=str,
                    help='path to store predictions')
parser.add_argument('metric_file', type=str,
                    help='path to store metrics')                    
parser.add_argument('--debug', type=bool, default=False,
                    help='set to True to show logging')



# Hint: You might find it helpful to define functions 
# that do the following:
# 1. Calculate Alphas
# 2. Calculate Betas
# 3. Implement the LogSumExpTrick
# 4. Calculate probabilities and predictions

def lse_trick(v):
    m = np.max(v)
    return m + np.log(np.sum(np.exp(v - m)))

def calc_alphas(words, word_dict, init, emit, trans):
    alphas = np.zeros((len(words), init.shape[0]))
    for i in range(len(words)):
        if i == 0:
            alphas[0, :] = np.log(init) + np.log(emit[:, word_dict[words[0]]])
        else:
            alphas[i, :] = np.log(emit[:, word_dict[words[i]]])
            for j in range(init.shape[0]):
                alphas[i, j] += lse_trick(alphas[i-1, :] + np.log(trans[:, j]))
    return alphas

def calc_betas(words, word_dict, emit, trans):
    N = len(words)
    betas = np.zeros((N, emit.shape[0]))
    for i in range(N):
        if i == 0:
            betas[N-1, :] = np.zeros(emit.shape[0])
        else:
            for j in range(emit.shape[0]):
                betas[N-1-i, j] = lse_trick(np.log(emit[:, word_dict[words[N-i]]]) + betas[N-i, :] + np.log(trans[j, :]))
    return betas

# TODO: Complete the main function
def main(args):

    # Get the dictionaries
    word_dict = make_dict(args.index_to_word)
    tag_dict = make_dict(args.index_to_tag)
    tag_idx = {v:k for k,v in tag_dict.items()}

    # Parse the validation file
    sentences, tags = parse_file(args.validation_input)

    ## Load your learned matrices
    ## Make sure you have them in the right orientation
    ## TODO:  Uncomment the following line when you're ready!
    
    init, emission, transition = get_matrices(args)

    
    predicted_tags = []
    ll_sum = 0
    for i in range(len(sentences)):
        alphas = calc_alphas(sentences[i], word_dict, init, emission, transition) 
        ll_sum += lse_trick(alphas[alphas.shape[0]-1, :])
        betas = calc_betas(sentences[i], word_dict, emission, transition)
        totals = alphas + betas
        predict_tag_idx = np.argmax(totals, axis=1)
        predict_tag = []
        for j in range(predict_tag_idx.shape[0]):
            predict_tag.append(tag_idx[predict_tag_idx[j]])
        predicted_tags.append(predict_tag)
    avg_log_likelihood = ll_sum / len(sentences)
    
    accuracy = 0 # We'll calculate this for you

    ## Writing results to the corresponding files.  
    ## We're doing this for you :)
    ## TODO: Just Uncomment the following lines when you're ready!

    accuracy = write_predictions(args.prediction_file, sentences, predicted_tags, tags)
    write_metrics(args.metric_file, avg_log_likelihood, accuracy)

    return

if __name__ == "__main__":
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(format='[%(asctime)s] {%(pathname)s:%(funcName)s:%(lineno)04d} %(levelname)s - %(message)s',
                            datefmt="%H:%M:%S",
                            level=logging.DEBUG)
    logging.debug('*** Debugging Mode ***')
    main(args)
