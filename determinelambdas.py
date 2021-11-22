#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------------------
#The purpose of this file is to apply the Viterbi algorithm for predicting the POS tags for each
#sentence in the test corpus
#
#Copyright (C) 2021, released under MIT License
#Author: Raihan Ahmed, Chicago, IL
#email: rahmed10@neiu.edu
#-------------------------------------------------------------------------------------------

import math
import time
import pickle


import math
import time
import pickle

LOG_ZERO = -1000

def transition_probs(taglists, unigrams, bigrams, trigrams):
    unigram_total = sum(unigrams.values())
    unigram_p = {(a,): math.log(unigrams[(a,)], 2) - math.log(unigram_total, 2) for a, in unigrams}

    unigrams[START_SYMBOL] = len(taglists)
    bigram_p = {(a, b): math.log(bigrams[(a, b)], 2) - math.log(unigrams[(a,)], 2) for a, b in bigrams}

    bigrams[(START_SYMBOL, START_SYMBOL)] = len(taglists)
    trigram_p = {(a, b, c): math.log(trigrams[(a, b, c)], 2) - math.log(bigrams[(a, b)], 2) for a, b, c in trigrams}

    #will need to experimentally determine lambda values by applying the viterbi algorithm in a for loop to maximize accuracy
    trigram_d = {(a, b , c): math.log(0.3 * (2**trigram_p[(a, b, c)]) + 0.4 * (2**bigram_p[(b, c)]) + 0.3 * (2**unigram_p[(c,)]), 2) for a, b, c in trigram_p}

    return unigram_p, bigram_p, trigram_p, trigram_d

def viterbi():














if __name__ == '__main__':
