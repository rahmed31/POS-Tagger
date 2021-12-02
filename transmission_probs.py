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
#order ngrams. These hyperparameters are experimentally determined in this file such that the accuracy of the POS tagger is
#maximized.
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
from accuracy import calculate_accuracy
from train_model import clean_text

output_path = 'data/model_data/'
START_SYMBOL = '*'

def lambda_candidates(start, end, step):
    """ Function that generates all possible sets of lambda values that sum to 1 for deleted interpolation algorithm.
        Runtime complexity: O(n^3)"""

    l = np.arange(start, end, step)

    result = []

    for i in range(len(l)):
        for j in range(len(l)):
            for k in range(len(l)):
                if (l[i] + l[j] + l[k] == 1):
                    result.append([l[i], l[j], l[k]])

    return result

#Need to fix argument inputs for this function
def find_lambdas(candidate_values, taglists, pos_set, test_set, test_tags, unigrams, bigrams, trigrams, e_probs):
    """ Function to experimentally determine the hyperparameters (i.e., lambda values) for the deleted interpolation
        algorithm. It returns a list containing the lambda values and a dictionary containing the log transition probabilities
        for each POS trigram present in the training corpus -- those of which have maximized the accuracy of the Viterbi algorithm.
        This function specifically caters to files with the format of the Brown corpus. Amortized runtime complexity: O(n) """

    max_accuracy = 0
    lambda_values = []
    q_probs = {}

    for lambdas in candidate_values:
        log_values = transition_probs(taglists, unigrams, bigrams, trigrams, lambdas)
        tagged_sentences = viterbi_algorithm(test_set, pos_set, known_words, q_probs, e_probs)

        model_tags = [[wordtag.rsplit('/', 1)[-1] for wordtag in line.strip().split(" ")] for line in tagged_sentences]
        accuracy = calculate_accuracy(test_tags, model_tags)

        if accuracy > max_accuracy:
            max_accuracy = accuracy
            lambda_values = lamdas
            q_probs = log_values

    return lambda_values, q_probs

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
    q_probs = {(a, b , c): math.log(lambdas[2] * (2**trigram_p[(a, b, c)]) + lambdas[1] * (2**bigram_p[(b, c)]) + lambdas[0] * (2**unigram_p[(c,)]), 2) for a, b, c in trigram_p}

    return q_probs

if __name__ == '__main__':

    start = time.perf_counter()

    unigrams = dict(pickle.load(open(output_path + "unigrams.pickle", "rb" )))
    bigrams = dict(pickle.load(open(output_path + "bigrams.pickle", "rb" )))
    trigrams = dict(pickle.load(open(output_path + "trigrams.pickle", "rb" )))

    taglists = pickle.load(open(output_path + "taglists.pickle", "rb" ))
    e_probs = dict(pickle.load(open(output_path + "e_probs.pickle", "rb" )))
    known_words = pickle.load(open(output_path + "known_words.pickle", "rb" ))
    pos_set = pickle.load(open(output_path + "pos_set.pickle", "rb" ))

    test_set, test_tags = clean_text('data/test_corpus.txt')

    #DOES NOT NEED TO BE RUN AGAIN
    # candidate_values = lambda_candidates(0.001, 1, 0.001)

    # candidate_values = pickle.dump(candidate_values, open(output_path + "candidate_values.pickle", "wb"))
    candidate_values = pickle.load(open(output_path + "candidate_values.pickle", "rb"))
    # print(len(candidate_values))

    #NOT GOING TO USE THIS FUNCTION AT THE MOMENT
    # lambda_values, q_probs = find_lambdas(candidate_values, taglists, pos_set, test_set, test_tags, unigrams, bigrams, trigrams, e_probs)
    # print(q_probs)

    #found lambda values online because viterbi algorithm doesn't work yet
    q_probs = transition_probs(taglists, unigrams, bigrams, trigrams, [0.125, 0.394, 0.481])
    # print(q_probs)

    q_probs = pickle.dump(q_probs, open(output_path + "q_probs.pickle", "wb" ))
    # lambda_values = pickle.dump(lambda_values, open(output_path + "lamnda_values.pickle", "wb" ))

    finish = time.perf_counter()
    print(f'Finished in {round(finish-start, 2)} second(s)')
