#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------------------
#The purpose of this file is to calculate the probabilities of all unigrams, bigrams, and trigrams
#that occur in the training corpus, as well as emission probabilities for each word/tag pair. These
#probabilities allow our model to predict the POS tags for the test corpus. To remedy the issue of
#unseen trigrams or unrecognized words in the test corpus, interpolation is necessary to approximate
#the probability of each POS tag sequence when assigning them. The deleted interpolation algorithm is
#used to achieve this, along with generalizing low frequency words in the training corpus
#where count < 5 so that unknown words that appear in the test corpus (which will also be generalized)
#can still be predicted with a non zero probability.
#
#
#Copyright (C) 2021, released under MIT License
#Author: Raihan Ahmed, Chicago, IL
#email: rahmed10@neiu.edu
#-------------------------------------------------------------------------------------------

import re
import os
import pickle
import math
import time
import numpy as np
from collections import defaultdict

output_path = 'data/model_data/'
RARE_SYMBOL = '_RARE_'
START_SYMBOL = '*'
MAX_FREQ_RARE = 5

#function for retrieving high frequency words from training corpus
def high_freq(tokenlists):
    """ This function is used to retrieve high frequency words from the training corpus to be later
        used when applying to Viterbi algorithm for POS tagging on the test corpus. """

    known_words = set()
    word_count = defaultdict(int)

    for tokenlist in tokenlists:
        for token in tokenlist:
            word_count[token] += 1

    for token, count in word_count.items():
        if count >= MAX_FREQ_RARE:
            known_words.add(token)

    return known_words

#helper function for subcategorizing rare words in the training corpus
def replace_rare(tokenlists, known_words):
    """ Function to replace (a.k.a "generalize") low frequency words that appear in the training
        corpus. """

    for i, tokenlist in enumerate(tokenlists):
        for j, token in enumerate(tokenlist):
            if token not in known_words:
                tokenlists[i][j] = morphosyntactic_subcategorize(token)

    return tokenlists

#Not sure how accurate this will be...
def morphosyntactic_subcategorize(word):
    if not re.search(r'\w', word):
        return '_PUNCS_'
    elif re.search(r'[A-Z]', word):
        return '_CAPITAL_'
    elif re.search(r'\d', word):
        return '_NUM_'
    elif re.search(r'(ion\b|ity\b|ics\b|ment\b|ence\b|ent\b|ant\b|ance\b|ness\b|ist\b|ee\b|ism\b|or\b|ship\b)', word):
        return '_NOUN_'
    elif re.search(r'(ate\b|fy\b|en\b|ize\b|ing\b|\ben|\bem|\bre|\bdis|\bre|\bmis|\binter|\bsub|ed\b)', word):
        return '_VERB_'
    elif re.search(r'(\bun|\bin|ble\b|ry\b|ish\b|\bdis|\bir|\bil|ous\b|ical\b|\bnon|ent\b|ive\b|able\b|ful\b)',word):
        return '_ADJ_'
    elif re.search(r'(ly\b|ally\b|ily\b|wise\b|wards\b)',word):
        return '_ADV_'
    else:
        return RARE_SYMBOL

def emission_probs(tokenlists, taglists):
    """ Function to find emission counts for each word/tag pair after replacing low frequency words with their
        generalized form. It returns a dictionary containing the corpus' emission probabilities for each token/tag tuple.
        This function specifically caters to files with the format of the Brown corpus. Runtime complexity: O(n^2) """

    e_values_c = defaultdict(int)
    tag_c = defaultdict(int)

    for sent_words, sent_tags in zip(tokenlists, taglists):
        for word, tag in zip(sent_words, sent_tags):
            e_values_c[(word, tag)] += 1
            tag_c[tag] += 1

    e_values = {(word, tag): math.log(e_values_c[(word, tag)], 2) - math.log(tag_c[tag], 2) for word, tag in e_values_c}
    tagset = set(tag_c)

    return e_values, tagset

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

if __name__ == '__main__':

    start = time.perf_counter()

    unigrams = dict(pickle.load(open(output_path + "unigrams.pickle", "rb" )))
    bigrams = dict(pickle.load(open(output_path + "bigrams.pickle", "rb" )))
    trigrams = dict(pickle.load(open(output_path + "trigrams.pickle", "rb" )))

    tokenlists = pickle.load(open(output_path + "tokenlists.pickle", "rb" ))
    taglists = pickle.load(open(output_path + "taglists.pickle", "rb" ))

    #getting high frequency and low frequency words from training corpus
    known_words = high_freq(tokenlists)
    known_words = pickle.dump(known_words, open(output_path + "known_words.pickle", "wb" ))
    #print(known_words)

    #replacing low frequency words that appear in the training corpus with their generalized form
    tokenlists = replace_rare(tokenlists, known_words)
    # print(tokenlists)

    #find emission probabilities for each word/tag pair, and retreive a set containing all possible tags
    #for this dataset to be later used by the viterbi algorithm
    e_probs, pos_set = emission_probs(tokenlists, taglists)
    # print(e_probs)

    e_probs = pickle.dump(e_probs, open(output_path + "e_probs.pickle", "wb" ))
    pos_set = pickle.dump(pos_set, open(output_path + "pos_set.pickle", "wb" ))

    # #find transition probabilities for each trigram; to be later used by the viterbi algorithm
    # q_probs = transition_probs(taglists, unigrams, bigrams, trigrams)
    # # print(q_probs)
    # q_probs = pickle.dump(q_probs, open(output_path + "q_probs.pickle", "wb" ))

    finish = time.perf_counter()
    print(f'Finished in {round(finish-start, 2)} second(s)')

###############################################################################################################
