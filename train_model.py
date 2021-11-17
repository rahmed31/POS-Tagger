#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------------------
#The purpose of this code is to provide the model output that's created when using the Brown training
#corpus (data/train_corpus.txt) to train a Part of Speech (POS) tagger. The model is written into a
#text file, which is subsequently saved to the data folder as model_file.txt. The purpose of the text file
#is to provide an easy visualization of what a Trigram HMM model could look like when creating your own POS tagger!
#Instead of using model_file.txt in further applications, the unigram, bigram, trigram, and emission dictionaries
#will be pickled to provide quick and easy access for the POS tagger.
#
#
#(C) 2021 Raihan Ahmed, Chicago, IL
#Released under MIT License
#email: rahmed10@neiu.edu
#-------------------------------------------------------------------------------------------

import sys
import re
import os
import pickle

START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'

#get ngrams in tuple form
def pos_ngram(taglists, n):
    """ Function to find pos ngrams of length 'n'. It returns a dictionary containing the corpus'
        pos sequences in tuple form and their frequencies. This function specifically caters to files
        with the format of the Brown corpus. Runtime complexity: O(n^2) """

    tag_sequences = {}

    for taglist in taglists:
        #add n - 1 star symbols to the beginning of the list to account for ngrams of different size
        taglist = ((n - 1) * [START_SYMBOL] + taglist + [STOP_SYMBOL])

        for i in range(len(taglist) - n + 1):
            #creating ngram tuple from taglist
            ngram  = tuple(taglist[i:i+n])

            if ngram in tag_sequences:
                tag_sequences[ngram] += 1
            else:
                tag_sequences.update({ngram : 1})

    #sort tag_sequences dictionary from greatest count to lowest count
    sorted_tag_sequences = sorted(tag_sequences.items(), key=lambda x: x[1], reverse = True)

    return sorted_tag_sequences

def emissions(tokenlists, taglists):
    """ Function to find emission counts for each word and their associated POS tag. It returns a dictionary
        containing the corpus' emission counts for each token/tag tuple. This function specifically caters to files
        with the format of the Brown corpus. Runtime complexity: O(n^2) """

    emissions = {}

    for tokenlist, taglist in zip(tokenlists, taglists):
        for token, tag in zip(tokenlist, taglist):
            if (token, tag) in emissions:
                emissions[(token, tag)] += 1
            else:
                emissions.update({(token, tag) : 1})

    #sort tag_sequences dictionary from greatest count to lowest count
    sortedemissions = sorted(emissions.items(), key=lambda x: x[1], reverse = True)

    return sortedemissions

def clean_text(training_corpus):
    """ Function used for cleaning text from training data that follows the format of the Brown corpus. Closed
        category words and punctuation are not removed to be able to ensure that training sentences are
        syntactically sound. The first return value is a list that contains distinct lists holding the tokens
        for each sentence in the training corpus. The second return value is a list that contains distinct
        lists holding the parts of speech present for each sentence in the training corpus. """

    try:
        #read text file
        with open(training_corpus, 'r') as f:
            #read all input from file at once
            lines = f.read()

    except IOError:
        print("Error: The input file does not appear to exist! Operation terminated.")
    else:
        #Split lines by new line character, remove empty lines, lowercase everything
        lines = [line.lower().strip() for line in lines.splitlines() if len(line.strip()) != 0]

        #Remove extra spaces between words
        lines = [re.sub(' +', ' ', line) for line in lines]

        tokenlists = [[wordtag.rsplit('/', 1)[0] for wordtag in line.strip().split(" ")] for line in lines]

        taglists = [[wordtag.rsplit('/', 1)[-1] for wordtag in line.strip().split(" ")] for line in lines]

        f.close()

        return tokenlists, taglists

def create_model_file(tokenlists, taglists):
    """ Function for creating trigram hidden Markov model using POS unigrams, bigrams, trigrams, and
        word/tag pairs with their emissions. The model is written onto a file in the library. The text file
        will not be used beyond this point. The purpose of the model text file """

    tag_size = 0
    tag_total = 0

    with open('data/model_data/model_file.txt', 'w+') as f:
        #clear file if contents already exist
        f.truncate(0)

        f.write('@unigrams@' + '\n')
        for key, value in pos_ngram(taglists, 1):
            tag_size += 1
            tag_total += value
            string = str(key) + '\t' + str(value)
            f.write(string + '\n')

        f.write('@bigrams@' + '\n')
        for key, value in pos_ngram(taglists, 2):
            string = str(key) + '\t' + str(value)
            f.write(string + '\n')

        f.write('@trigrams@' + '\n')
        for key, value in pos_ngram(taglists, 3):
            string = str(key) + '\t' + str(value)
            f.write(string + '\n')

        f.write('@emissions@' + '\n')
        for key, value in emissions(tokenlists, taglists):
            string = str(key) + '\t' + str(value)
            f.write(string + '\n')

        #tag_size is not necessarily the same as vocab size, but tag_total will always
        #be equal to total words
        f.write("@tag_total@\t" + str(tag_total) + '\n')
        f.write("@tag_size@\t" + str(tag_size))

        f.close()

    print("--------------------------------")
    print("Model file successfully created!")
    print("--------------------------------")

#Main driver used for debugging
if __name__ == '__main__':

    tokenlists, taglists = clean_text('data/train_corpus.txt')

    create_model_file(tokenlists, taglists)

    unigrams = pos_ngram(taglists, 1)
    bigrams = pos_ngram(taglists, 2)
    trigrams = pos_ngram(taglists, 3)
    emissions = emissions(tokenlists, taglists)

    pickle.dump(unigrams, open("data/model_data/unigrams.pickle", "wb" ))
    pickle.dump(bigrams, open("data/model_data/bigrams.pickle", "wb" ))
    pickle.dump(trigrams, open("data/model_data/trigrams.pickle", "wb" ))
    pickle.dump(emissions, open("data/model_data/emissions.pickle", "wb" ))

##########################################################################################################
