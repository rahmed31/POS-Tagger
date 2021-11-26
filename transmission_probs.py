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
import numpy as np
from viterbi import viterbi_algorithm
from emission_probs import replace_rare
from train_model import clean_text

output_path = 'data/model_data/'

def lambda_candidates(start, end, step):
    """ Function that generates all possible sets of lambda values that sum to 1 for deleted interpolation algorithm.
        Runtime complexity: O(n^3)"""

    l = np.arange(start, end, step)

    result = []

    for i in range(len(l)):
        for j in range(len(l)):
            for k in range(len(l)):
                if (l[i] + l[j] + l[k] == 1):
                    result.append(tuple((l[i], l[j], l[k])))

    return result

def find_lambdas(candidate_values, test_set, taglists, unigrams, bigrams, trigrams, e_values):
    """ Function to experimentally determine the hyperparameters (i.e., lambda values) for the deleted interpolation
        algorithm. This function utilizes a 10-fold cross validation technique to determine which lambda values maximize
        the accuracy of the viterbi algorithm when determining the POS tags for the validation sets. It returns a list containing
        the lambda values and a dictionary containing the log transition probabilities for each POS trigram present in the training
        corpus. This function specifically caters to files with the format of the Brown corpus. Amortized runtime complexity: O(n) """

    max_accuracy = 0
    lambda_values = []
    q_values = {}

    for lambdas in candidate_values:
        log_values = transition_probs(taglists, unigrams, bigrams, trigrams, lambdas)
        tags = viterbi_algorithm(test_set, taglist, known_words, q_values, e_values)
        accuracy = calculate_accuracy(tags, test_set)

        if accuracy > max_accuracy:
            max_accuracy = accuracy
            lambda_values = lamdas
            q_values = log_values

    return lambda_values, q_values

#might need to modify this function
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
    q_values = {(a, b , c): math.log(lambdas[0] * (2**trigram_p[(a, b, c)]) + lambdas[1] * (2**bigram_p[(b, c)]) + lambdas[2] * (2**unigram_p[(c,)]), 2) for a, b, c in trigram_p}

    return q_values

#INCOMPLETE FUNCTION
# def calculate_accuracy(tags, test_set):
    """ Function to calculate the accuracy of the Viterbi algorithm by comparing the output of the POS tagger to the actual tags
        provided in each validation set. """




        # return accuracy

if __name__ == '__main__':

    start = time.perf_counter()

    unigrams = dict(pickle.load(open(output_path + "unigrams.pickle", "rb" )))
    bigrams = dict(pickle.load(open(output_path + "bigrams.pickle", "rb" )))
    trigrams = dict(pickle.load(open(output_path + "trigrams.pickle", "rb" )))

    tokenlists = pickle.load(open(output_path + "tokenlists.pickle", "rb" ))
    taglists = pickle.load(open(output_path + "taglists.pickle", "rb" ))
    e_values = dict(pickle.load(open(output_path + "e_probs.pickle", "rb" )))
    known_words = pickle.load(open(output_path + "known_words.pickle", "rb" ))
    pos_set = pickle.load(open(output_path + "pos_set.pickle", "rb" ))

    # candidate_values = lambda_candidates(0.001, 1, 0.001)
    # candidate_values = pickle.dump(candidate_values, open(output_path + "candidate_values.pickle", "wb"))
    candidate_values = pickle.load(open(output_path + "candidate_values.pickle", "rb"))

    # lambda_values, q_values = find_lambdas(candidate_values, taglists, unigrams, bigrams, trigrams, e_values)
    # print(q_probs)

    # q_probs = pickle.dump(q_probs, open(output_path + "q_probs.pickle", "wb" ))
    # lambda_values = pickle.dump(lambda_values, open(output_path + "lamnda_values.pickle", "wb" ))

    finish = time.perf_counter()
    print(f'Finished in {round(finish-start, 2)} second(s)')
