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

#this method is called by transition_probs() to approximate probabilities for unseen trigrams
#needs fix
def deleted_interpolation(unigrams, bigrams, trigrams):
    lambda1 = 0
    lambda2 = 0
    lambda3 = 0

    for a, b, c in trigrams.keys():
        v = trigrams[(a, b, c)]

        if v > 0:

            try:
                c1 = (v - 1) / (bigrams[(a, b)] - 1)
            except:
                c1 = 0
            try:
                c2 = (bigrams[(a, b)] - 1) / (unigrams[(a,)] - 1)
            except:
                c2 = 0
            try:
                c3 = (unigrams[(a,)] - 1) / (sum(unigrams.values()) - 1)
            except:
                c3 = 0

            clist = [c1, c2, c3]
            m = np.argmax(clist)

            if m == 0:
                lambda3 += v
            if m == 1:
                lambda2 += v
            if m == 2:
                lambda3 += v

    weights = [lambda1, lambda2, lambda3]
    weights = [a / sum(weights) for a in weights]

    return weights

def viterbi():














if __name__ == '__main__':
