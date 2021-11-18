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


output_path = 'data/model_data/'
MAX_FREQ_RARE = 4
RARE_SYMBOL = '_RARE_'

#calculate emissions probabilities after executing replace_rare() functions
def emission_probs():


#does not depend on emission_probs() or replace_rare() functions
def transition_probs():


#this method is called by transition_probs() to approximate probabilities for unseen trigrams
def deleted_interpolation():


#helper method for subcategorizing rare words in emissions dictionary
def replace_rare():



def subcategorize(word):
    if not re.search(r'\w', word):
        return '_PUNCS_'
    elif re.search(r'[A-Z]', word):
        return '_CAPITAL_'
    elif re.search(r'\d', word):
        return '_NUM_'
    elif re.search(r'(ion\b|ity\b|ics\b|ment\b|ence\b|ent\b|ant\b|ance\b|ness\b|ist\b|ee\b|ism\b|ship\b)', word):
        return '_NOUN_'
    elif re.search(r'(ate\b|fy\b|en\b|ize\b|\ben|\bem|\bre|\bdis|\bre|\bmis|\binter|\bsub|\bbe)', word):
        return '_VERB_'
    elif re.search(r'(\bun|\bin|ble\b|ry\b|ish\b|\bdis|\bir|\bil|ous\b|ical\b|\bnon|ent\b|ive\b|ful\b)',word):
        return '_ADJ_'
    elif re.search(r'(ly\b|ally\b|ily\b|wise\b|wards\b)',word):
        return '_ADV_'
    else:
        return RARE_SYMBOL


if __name__ == '__main__':
    df = pickle.load(open('lib/df_processed.pickle', "rb" ))

    unigrams = pickle.load(open(output_path + "unigrams.pickle", "wb" ))
    bigrams = pickle.load(open(output_path + "bigrams.pickle", "wb" ))
    trigrams = pickle.load(open(output_path + "trigrams.pickle", "wb" ))
    emissions = pickle.load(open(output_path + "emissions.pickle", "wb" ))

    tokenlists = pickle.load(open(output_path + "tokenlists.pickle", "wb" ))
    taglists = pickle.load(open(output_path + "taglists.pickle", "wb" ))


###############################################################################################################
