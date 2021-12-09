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
POS_SET = set()

#Error with algorithm, seems like pi and bp dictionaries aren't updating with correctly
def viterbi_algorithm(test_sentences, known_words, q_probs, e_probs):
    """ Applying the Viterbi algorithm with time complexity O(n*k^2) """

    tagged = []

    # Define tagsets S(k)
    def S(k, token = None):
        if k in (-1, 0):
            return {START_SYMBOL}
        elif token == None and k > 0:
            return POS_SET
        else:
            tags = set()

            for (a, b) in e_probs.keys():
                if (token == a):
                    tags.add(b)
                    POS_SET.add(b)

            return tags

    def reset():
        global POS_SET
        POS_SET.clear()

    for original_sentence in test_sentences:

        sent_words = [word if word in known_words else morphosyntactic_subcategorize(word) for word in original_sentence]
        n = len(sent_words)

        # pi[(k, u, v)]: max probability of a tag sequence ending in tags u, v at position k
        pi = defaultdict(float)
        # initialize pi
        pi[(0, START_SYMBOL, START_SYMBOL)] = 0.0
        # bp[(k, u, v)]: backpointers to recover the argmax of pi[(k, u, v)]
        bp = {}

        for k in range(1, n+1):

            U = S(k-1, sent_words[k-1])
            V = S(k, sent_words[k-1])
            W = S(k-2, sent_words[k-1])

            for u in U:
                for v in V:
                    prob = defaultdict(float)
                    for w in W:

                        max_score = float('-Inf')
                        max_tag = None

                        if e_probs.get((sent_words[k-1], v), 0) != 0:
                            score = pi[(k-1, w, u)] + q_probs.get((w, u, v), LOG_ZERO) + e_probs.get((sent_words[k-1], v))

                            if score > max_score:
                                max_score = score
                                max_tag = w

                    pi[(k, u, v)] = max_score
                    bp[(k, u, v)] = max_tag

                    # print(pi[(k, u, v)])
                    # print(bp[(k, u, v)])
                    # print(k, u, v)
                    # print(sent_words[k-1])

        max_score = float('-Inf')
        u_max = None
        v_max = None

        U = S(n-1)
        V = S(n)

        for u in U:
            for v in V:
                score = pi.get((n, u, v), LOG_ZERO) + q_probs.get((u, v, STOP_SYMBOL), LOG_ZERO)
                if score > max_score:
                    max_score = score
                    u_max = u
                    v_max = v

        # print(u_max, v_max)

        # tags = deque()
        # tags.append(v_max)
        # tags.append(u_max)

        # for i, k in enumerate(range(n-2, 0, -1)):
        #     tags.append(bp[(k+2, tags[i+1], tags[i])])
        # tags.reverse()

        tags = [" "]*n
        tags[n-2], tags[n-1] = u, v

        for k in range(n-2, 1, -1):
            tags[k] = bp[(k+2, tags[k+1], tags[k])]

        tagged_sentence = deque()
        for j in range(0, n):
            tagged_sentence.append(original_sentence[j] + '/' + tags[j])
        tagged_sentence.append('\n')
        tagged.append(' '.join(tagged_sentence))

        reset()

    return tagged

if __name__ == '__main__':

    start = time.perf_counter()

    q_probs = dict(pickle.load(open(output_path + "q_probs.pickle", "rb" )))
    e_probs = dict(pickle.load(open(output_path + "e_probs.pickle", "rb" )))
    known_words = pickle.load(open(output_path + "known_words.pickle", "rb" ))
    # pos_set = pickle.load(open(output_path + "pos_set.pickle", "rb" ))

    test_sentences, tags = clean_text('data/test_corpus.txt')
    # print(e_probs)
    # print(pos_set)
    # print(len(pos_set))
    # print(q_probs)
    # print(sorted(q_probs.items(), key=lambda x: x[1]))

    tagged_sentences = viterbi_algorithm(test_sentences, known_words, q_probs, e_probs)
    # print(tagged_sentences)

    # tagged_sentences = pickle.dump(tagged, open(output_path + "tagged_sentences.pickle", "wb"))

    finish = time.perf_counter()
    print(f'Finished in {round(finish-start, 2)} second(s)')
