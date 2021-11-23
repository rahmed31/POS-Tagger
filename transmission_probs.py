#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------------------
#The purpose of this file is to calculate the transmission probabilities of all POS trigrams that occur in
#the training corpus. These probabilities allow our model to predict the POS tags for the test corpus.
#To remedy the issue of unseen POS trigrams that may result from the test corpus, interpolation is necessary to approximate
#the probability of each POS trigram when calculating them. The deleted interpolation algorithm is
#used to achieve this so that unknown POS tag sequences that may appear in the test corpus can still be predicted
#with a non zero probability using lower order ngrams. Deleted interpolation also requires the use of lambda
#values (i.e., lambda1, lambda2, lambda3) to help with maximizing the accuracy of the POS tagger when utilizing lower
#order ngrams. These hyperparameters will be experimentally determined in this file such that the accuracy of the POS tagger is
#maximized when using 10-fold cross validation on the training corpus.
#
#Copyright (C) 2021, released under MIT License
#Author: Raihan Ahmed, Chicago, IL
#email: rahmed10@neiu.edu
#-------------------------------------------------------------------------------------------

import math
import time
import pickle
from viterbi import viterbi_algorithm
from interpolations import replace_rare

output_path = 'data/model_data/'

#INCOMPLETE FUNCTION
def find_lambdas(taglists, unigrams, bigrams, trigrams):
    """ Function to experimentally determine the hyperparameters (i.e., lambda values) for the deleted interpolation
        algorithm. This function utilizes a 10-fold cross validation technique to determine which lambda values maximize
        the accuracy of the viterbi algorithm when determining the POS tags for the validation sets. It returns a list containing
        the lambda values and a dictionary containing the log transition probabilities for each POS trigram present in the training
        corpus. This function specifically caters to files with the format of the Brown corpus. Runtime complexity: O(n^2) """

    max_accuracy = 0
    lambda_values = []
    q_values = {}

    for i in range(0.001, 1, 0.001):
        for j in range(0.001):
            for k in range():
                log_values = transition_probs(taglists, unigrams, bigrams, trigrams, [i, j, k])

                # tags = viterbi_algorithm(brown_dev_words, taglist, known_words, q_values, e_values)
                #
                # accuracy = calculate_accuracy(tags, brown_dev_words)

                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    lambda_values = [i, j, k]
                    q_values = log_values

    return lambda_values, q_values

#INCOMPLETE FUNCTION
# def calculate_accuracy(tags, brown_dev_words):
    """ Function to calculate the accuracy of the Viterbi algorithm by comparing the output of the POS tagger to the actual tags
        provided in each validation set. """




        # return accuracy

def transition_probs(taglists, unigrams, bigrams, trigrams, lambdas):
    """ Function to find log transition probabilities for each POS trigram in the training corpus using deleted interpolation.
        It returns a dictionary containing the corpus' log transition probabilities for each POS trigram present in the training corpus.
        This function specifically caters to files with the format of the Brown corpus. Runtime complexity: O(n^2) """

    unigram_total = sum(unigrams.values())
    unigram_p = {(a,): math.log(unigrams[(a,)], 2) - math.log(unigram_total, 2) for a, in unigrams}

    unigrams[START_SYMBOL] = len(taglists)
    bigram_p = {(a, b): math.log(bigrams[(a, b)], 2) - math.log(unigrams[(a,)], 2) for a, b in bigrams}

    bigrams[(START_SYMBOL, START_SYMBOL)] = len(taglists)
    trigram_p = {(a, b, c): math.log(trigrams[(a, b, c)], 2) - math.log(bigrams[(a, b)], 2) for a, b, c in trigrams}

    #calculating log transmission probabilities
    q_values = {(a, b , c): math.log(lambdas[0] * (2**trigram_p[(a, b, c)]) + lambda[1] * (2**bigram_p[(b, c)]) + lambda[2] * (2**unigram_p[(c,)]), 2) for a, b, c in trigram_p}

    return q_values

if __name__ == '__main__':

    start = time.perf_counter()

    unigrams = dict(pickle.load(open(output_path + "unigrams.pickle", "rb" )))
    bigrams = dict(pickle.load(open(output_path + "bigrams.pickle", "rb" )))
    trigrams = dict(pickle.load(open(output_path + "trigrams.pickle", "rb" )))

    tokenlists = pickle.load(open(output_path + "tokenlists.pickle", "rb" ))
    taglists = pickle.load(open(output_path + "taglists.pickle", "rb" ))
    e_values = pickle.load(open(output_path + "e_probs.pickle", "rb" ))
    known_words = pickle.load(open(output_path + "known_words.pickle", "rb" ))
    pos_set = pickle.load(open(output_path + "pos_set.pickle", "rb" ))

    lambda_values, q_values = find_lambdas(taglists, unigrams, bigrams, trigrams)
    # print(q_probs)

    # q_probs = pickle.dump(q_probs, open(output_path + "q_probs.pickle", "wb" ))

    finish = time.perf_counter()
    print(f'Finished in {round(finish-start, 2)} second(s)')
