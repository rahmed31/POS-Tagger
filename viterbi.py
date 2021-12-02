#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------------------
#The purpose of this file is to apply the Viterbi algorithm for predicting the POS tags for each
#sentence in the test corpus.
#
#Copyright (C) 2021, released under MIT License
#Author: Raihan Ahmed, Chicago, IL
#email: rahmed10@neiu.edu
#-------------------------------------------------------------------------------------------

import math
import time
import pickle
from train_model import clean_text
from collections import defaultdict, deque
from emission_probs import morphosyntactic_subcategorize

output_path = 'data/model_data/'
START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
LOG_ZERO = -1000

def viterbi_algorithm(test_sentences, pos_set, known_words, q_probs, e_probs):
    """ Applying the Viterbi algorithm with time complexity O(n*k^2) """

    tagged = []
    # pi[(k, u, v)]: max probability of a tag sequence ending in tags u, v at position k
    pi = defaultdict(float)
    # bp[(k, u, v)]: backpointers to recover the argmax of pi[(k, u, v)]
    bp = {}

    # Initialization
    pi[(0, START_SYMBOL, START_SYMBOL)] = 1.0

    # Define tagsets S(k)
    def S(k):
        if k in (-1, 0):
            return {START_SYMBOL}
        else:
            return pos_set

    # The Viterbi algorithm
    for original_sentence in test_sentences:
        sent_words = [word if word in known_words else morphosyntactic_subcategorize(word) for word in original_sentence]
        n = len(sent_words)

        for k in range(1, n):
            for u in S(k - 1):
                for v in S(k):

                    max_score = float('-Inf')
                    max_tag = None

                    for w in S(k - 2):
                        if e_probs.get((sent_words[k-1], v), 0) != 0:
                            score = pi.get((k-1, w, u), LOG_ZERO) + q_probs.get((w, u, v), LOG_ZERO) + e_probs.get((sent_words[k-1], v))
                            if score > max_score:
                                max_score = score
                                max_tag = w
                    pi[(k, u, v)] = max_score
                    bp[(k, u, v)] = max_tag

        max_score = float('-Inf')
        u_max, v_max = None, None

        for u in S(n - 1):
            for v in S(n):
                score = pi.get((n, u, v), LOG_ZERO) + q_probs.get((u, v, STOP_SYMBOL), LOG_ZERO)
                if score > max_score:
                    max_score = score
                    u_max = u
                    v_max = v

        tags = deque()
        tags.append(v_max)
        tags.append(u_max)

        #Error occurs at this stage:
        for i, k in enumerate(range(n-2, 0, -1)):
            tags.append(bp[(k+2, tags[i+1], tags[i])])
        tags.reverse()

        tagged_sentence = deque()
        for j in range(0, n):
            tagged_sentence.append(original_sentence[j] + '/' + tags[j])
        tagged_sentence.append('\n')
        tagged.append(' '.join(tagged_sentence))

    return tagged

if __name__ == '__main__':

    start = time.perf_counter()

    q_probs = dict(pickle.load(open(output_path + "q_probs.pickle", "rb" )))
    e_probs = dict(pickle.load(open(output_path + "e_probs.pickle", "rb" )))
    known_words = pickle.load(open(output_path + "known_words.pickle", "rb" ))
    pos_set = pickle.load(open(output_path + "pos_set.pickle", "rb" ))

    test_sentences, tags = clean_text('data/test_corpus.txt')
    # print(test_sentences)

    tagged = viterbi_algorithm(test_sentences, pos_set, known_words, q_probs, e_probs)
    # print(tagged)

    finish = time.perf_counter()
    print(f'Finished in {round(finish-start, 2)} second(s)')
