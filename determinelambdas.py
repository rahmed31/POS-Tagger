#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------------------
#The purpose of this file is to experimentally determine the lambda values that maximize the
#accurracy of the POS tagger.
#
#Copyright (C) 2021, released under MIT License
#Author: Raihan Ahmed, Chicago, IL
#email: rahmed10@neiu.edu
#-------------------------------------------------------------------------------------------

import math
import time
import pickle

output_path = 'data/model_data/'
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

def viterbi(brown_dev_words, taglist, known_words, q_values, e_values):
    tagged = []

    # pi[(k, u, v)]: max probability of a tag sequence ending in tags u, v at position k
    # bp[(k, u, v)]: backpointers to recover the argmax of pi[(k, u, v)]
    pi = defaultdict(float)
    bp = {}

    # Initialization
    pi[(0, START_SYMBOL, START_SYMBOL)] = 0.0

    # Define tagsets S(k)
    def S(k):
        if k in (-1, 0):
            return {START_SYMBOL}
        else:
            return taglist

    # The Viterbi algorithm
    for sent_words_actual in brown_dev_words:
        sent_words = [word if word in known_words else RARE_SYMBOL for word in sent_words_actual]
        n = len(sent_words)
        for k in range(1, n+1):
            for u in S(k-1):
                for v in S(k):
                    max_score = float('-Inf')
                    max_tag = None
                    for w in S(k - 2):
                        if e_values.get((sent_words[k-1], v), 0) != 0:
                            score = pi.get((k-1, w, u), LOG_PROB_OF_ZERO) + \
                                    q_values.get((w, u, v), LOG_PROB_OF_ZERO) + \
                                    e_values.get((sent_words[k-1], v))
                            if score > max_score:
                                max_score = score
                                max_tag = w
                    pi[(k, u, v)] = max_score
                    bp[(k, u, v)] = max_tag

        max_score = float('-Inf')
        u_max, v_max = None, None
        for u in S(n-1):
            for v in S(n):
                score = pi.get((n, u, v), LOG_PROB_OF_ZERO) + \
                        q_values.get((u, v, STOP_SYMBOL), LOG_PROB_OF_ZERO)
                if score > max_score:
                    max_score = score
                    u_max = u
                    v_max = v

        tag = deque()
        tags.append(v_max)
        tags.append(u_max)

        for i, k in enumerate(range(n-2, 0, -1)):
            tags.append(bp[(k+2, tags[i+1], tags[i])])
        tags.reverse()

        tagged_sentence = deque()
        for j in range(0, n):
            tagged_sentence.append(sent_words_actual[j] + '/' + tags[j])
        tagged_sentence.append('\n')
        tagged.append(' '.join(tagged_sentence))

    return tagged

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


    finish = time.perf_counter()
    print(f'Finished in {round(finish-start, 2)} second(s)')
