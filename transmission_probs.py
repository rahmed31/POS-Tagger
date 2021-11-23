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
from interpolations import replace_rare

output_path = 'data/model_data/'
LOG_ZERO = -1000

def find_lambdas(taglists, unigrams, bigrams, trigrams):
    max_accuracy = 0
    lambda_values = []
    q_values = {}

    for i in range(0.001, 1, 0.001):
        for j in range(0.001):
            for k in range():
                log_values = transition_probs(taglists, unigrams, bigrams, trigrams, [i, j, k])

                # tags = viterbi(brown_dev_words, taglist, known_words, q_values, e_values)
                #
                # accuracy = calculate_accuracy(tags, brown_dev_words)

                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    lambda_values = [i, j, k]
                    q_values = log_values

    return lambda_values, q_values

# def calculate_accuracy(tags, brown_dev_words):



def transition_probs(taglists, unigrams, bigrams, trigrams, lambdas):
    unigram_total = sum(unigrams.values())
    unigram_p = {(a,): math.log(unigrams[(a,)], 2) - math.log(unigram_total, 2) for a, in unigrams}

    unigrams[START_SYMBOL] = len(taglists)
    bigram_p = {(a, b): math.log(bigrams[(a, b)], 2) - math.log(unigrams[(a,)], 2) for a, b in bigrams}

    bigrams[(START_SYMBOL, START_SYMBOL)] = len(taglists)
    trigram_p = {(a, b, c): math.log(trigrams[(a, b, c)], 2) - math.log(bigrams[(a, b)], 2) for a, b, c in trigrams}

    #calculating log transmission probabilities
    q_values = {(a, b , c): math.log(lambdas[0] * (2**trigram_p[(a, b, c)]) + lambda[1] * (2**bigram_p[(b, c)]) + lambda[2] * (2**unigram_p[(c,)]), 2) for a, b, c in trigram_p}

    return q_values

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

    lambda_values, q_values = find_lambdas(taglists, unigrams, bigrams, trigrams)
    # print(q_probs)

    # q_probs = pickle.dump(q_probs, open(output_path + "q_probs.pickle", "wb" ))

    finish = time.perf_counter()
    print(f'Finished in {round(finish-start, 2)} second(s)')
