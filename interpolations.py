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
from collections import defaultdict
import time

output_path = 'data/model_data/'
RARE_SYMBOL = '_RARE_'
MAX_FREQ_RARE = 5
LOG_ZERO = -1000

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

#!!!NEED TO TEST THIS FUNCTION!!!
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

# def transition_probs():
#
#
# #this method is called by transition_probs() to approximate probabilities for unseen trigrams
# def deleted_interpolation():

if __name__ == '__main__':

    start = time.perf_counter()

    unigrams = dict(pickle.load(open(output_path + "unigrams.pickle", "rb" )))
    bigrams = dict(pickle.load(open(output_path + "bigrams.pickle", "rb" )))
    trigrams = dict(pickle.load(open(output_path + "trigrams.pickle", "rb" )))

    tokenlists = pickle.load(open(output_path + "tokenlists.pickle", "rb" ))
    taglists = pickle.load(open(output_path + "taglists.pickle", "rb" ))

    #getting high frequency and low frequency words from training corpus
    known_words = high_freq(tokenlists)
    #print(known_words)

    #replacing low frequency words that appear in the training corpus with their generalized form
    tokenlists = replace_rare(tokenlists, known_words)
    # print(tokenlists)

    #find emission probabilities for each word/tag pair, and retreive a set containing all possible tags
    #for this dataset to be later used by the viterbi algorithm
    e_probs, pos_set = emission_probs(tokenlists, taglists)

    e_probs = pickle.dump(e_probs, open(output_path + "e_probs.pickle", "wb" ))
    pos_set = pickle.dump(pos_set, open(output_path + "pos_set.pickle", "wb" ))


    finish = time.perf_counter()
    print(f'Finished in {round(finish-start, 2)} second(s)')

###############################################################################################################


# #helper function for subcategorizing rare words in the training corpus
# def replace_rare(tokenlists, known_words):
#     """ Function to replace (a.k.a "generalize") low frequency words that appear in the training
#         corpus. """
#
#     for i, tokenlist in enumerate(tokenlists):
#         for j, token in enumerate(tokenlist):
#             if token not in known_words:
#                 tokenlists[i][j] = subcategorize(token)
#
#     return tokenlists
#
#     # copy = emissions.copy()
#     #
#     # for (word, tag), value in copy.items():
#     #     if value < MAX_FREQ_RARE:
#     #         category = subcategorize(word)
#     #         new_key = category, tag
#     #         del emissions[(word, tag)]
#     #
#     #         if new_key in emissions:
#     #             emissions[new_key] += value
#     #         else:
#     #             emissions.update({new_key : value})
#     #
#     # return dict(sorted(emissions.items(), key=lambda x: x[1], reverse = True))


# #calculate emissions probabilities after executing replace_rare() functions
# def emission_probs(emissions, unigrams):
#     """ Function to calculate the emissions probabilities for each word/tag pair after applying
#         the replace_rare() function to generalize low frequency words in the training corpus. """
#
#     e_probs = {}
#
#     for (word, tag), value in emissions.items():
#         if not (word, tag) in e_probs:
#             e_probs[(word, tag)] = value/unigrams[(tag,)]
#
#     return e_probs



# #!!!NEED TO TEST THIS FUNCTION!!!
# def emission_probs(tokenlists, taglists):
#     """ Function to find emission counts for each word/tag pair after replacing low frequency words with their
#         generalized form. It returns a dictionary containing the corpus' emission probabilities for each token/tag tuple.
#         This function specifically caters to files with the format of the Brown corpus. Runtime complexity: O(n^2) """
#
#     e_values_c = defaultdict(int)
#     tag_c = defaultdict(int)
#
#     for sent_words, sent_tags in zip(tokenlists, taglists):
#         for word, tag in zip(sent_words, sent_tags):
#             e_values_c[(word, tag)] += 1
#             tag_c[tag] += 1
#
#     e_values = {(word, tag): math.log(e_values_c[(word, tag)], 2) - math.log(tag_c[tag], 2) for word, tag in e_values_c}
#     taglist = set(tag_c)
#
#     return e_values, taglist
#     # emissions = {}
#     # e_probs = {}
#     #
#     # for tokenlist, taglist in zip(tokenlists, taglists):
#     #     for token, tag in zip(tokenlist, taglist):
#     #         if (token, tag) in emissions:
#     #             emissions[(token, tag)] += 1
#     #         else:
#     #             emissions.update({(token, tag) : 1})
#     #
#     # for (word, tag), value in emissions.items():
#     #     if not (word, tag) in e_probs:
#     #         e_probs[(word, tag)] = value/unigrams[(tag,)]
#     #
#     # #sort e_probs dictionary from greatest count to lowest count
#     # e_probs = dict(sorted(e_probs.items(), key=lambda x: x[1], reverse = True))
#     #
#     # return e_probs
