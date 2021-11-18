#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------------------
#The purpose of this file is to calculate the probabilities of all unigrams, bigrams, and trigrams
#that occur in the training corpus, as well as emission probabilities for each word/tag pair. These
#probabilities allow our model to predict the POS tags for the test corpus. To remedy the issue of
#unseen trigrams in the test corpus, interpolation is necessary to approximate their probability of
#any given POS tag sequences when assigning them. The deleted interpolation algorithm is used to achieve this
#with a combination of taking into generalizing the low frequency words where count < 5.
#
#
#Copyright (C) 2021, released under MIT License
#Author: Raihan Ahmed, Chicago, IL
#email: rahmed10@neiu.edu
#-------------------------------------------------------------------------------------------

import re
import os
import pickle
from collections import defaultdict

output_path = 'data/model_data/'
MAX_FREQ_RARE = 5
RARE_SYMBOL = '_RARE_'

#calculate emissions probabilities after executing replace_rare() functions
def emission_probs(emissions, unigrams):
    e_probs = {}

    for (word, tag), value in emissions.items():
        if not (word, tag) in e_probs:
            e_probs[(word, tag)] = value/unigrams[(tag,)]

    return e_probs

# #does not depend on emission_probs() or replace_rare() functions
# def transition_probs():
#
#
# #this method is called by transition_probs() to approximate probabilities for unseen trigrams
# def deleted_interpolation():


#function for getting high frequency words from training corpus
def high_freq(tokenlists):
    known_words = set()
    word_c = defaultdict(int)

    for tokenlist in tokenlists:
        for token in tokenlist:
            word_c[token] += 1

    for token, count in word_c.items():
        if count >= MAX_FREQ_RARE:
            known_words.add(token)

    return known_words

#helper function for subcategorizing rare words and updating the emissions dictionary
def replace_rare(emissions):
    copy = emissions.copy()

    for (word, tag), value in copy.items():
        if value < MAX_FREQ_RARE:
            category = subcategorize(word)
            new_key = category, tag
            del emissions[(word, tag)]

            if new_key in emissions:
                emissions[new_key] += value
            else:
                emissions.update({new_key : value})

    return dict(sorted(emissions.items(), key=lambda x: x[1], reverse = True))

#Not sure how accurate this is going to be...
def subcategorize(word):
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

if __name__ == '__main__':

    unigrams = dict(pickle.load(open(output_path + "unigrams.pickle", "rb" )))
    bigrams = dict(pickle.load(open(output_path + "bigrams.pickle", "rb" )))
    trigrams = dict(pickle.load(open(output_path + "trigrams.pickle", "rb" )))
    emissions = dict(pickle.load(open(output_path + "emissions.pickle", "rb" )))

    tokenlists = pickle.load(open(output_path + "tokenlists.pickle", "rb" ))
    taglists = pickle.load(open(output_path + "taglists.pickle", "rb" ))

    #overwriting original emissions dictionary with updated/generalized emissions dictionary
    emissions = replace_rare(emissions)
    e_probs = emission_probs(emissions, unigrams)

    #getting high frequency and low frequency words from training corpus
    known_words = high_freq(tokenlists)

    # print(e_probs)

###############################################################################################################
