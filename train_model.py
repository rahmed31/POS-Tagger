#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
__author__ = "Raihan Ahmed"

import sys
import re
import os

def _pos_sequences(sentences, n):
    """ Function to find pos sequences of length 'n'. It returns a dictionary
        containing the corpus' pos sequences and their frequencies. This function specifically caters to files
        with the format of the Brown corpus. Runtime complexity: O(n^2) """

    tag_sequences = {}

    for sentence in sentences:
        #add n - 1 star symbols to the beginning of the list to account for ngrams of different size
        tags_list = ((n - 1) * ['*'])

        for word in sentence.split(" "):
            pos = word.rsplit('/', 1)[1]
            tags_list.append(pos)

        for i in range(len(tags_list) - n + 1):
            #joining pos tags in tags_list to create a single tag sequence
            sequence = ' '.join(tags_list[i:i+n])

            if sequence in tag_sequences:
                tag_sequences[sequence] += 1
            else:
                tag_sequences.update({sequence : 1})

    #sort tag_sequences dictionary from greatest count to lowest count
    sorted_tag_sequences = sorted(tag_sequences.items(), key=lambda x: x[1], reverse = True)

    return sorted_tag_sequences

def _emissions(sentences):
    """ Function to find emission counts for each word and their associated POS tag. It returns a dictionary
        containing the corpus' emission counts for each word. This function specifically caters to files
        with the format of the Brown corpus. Runtime complexity: O(n^2) """

    emissions = {}

    for sentence in sentences:

        for word in sentence.split(" "):

            if word in emissions:
                emissions[word] += 1
            else:
                emissions.update({word : 1})

    #sort tag_sequences dictionary from greatest count to lowest count
    sorted_emissions = sorted(emissions.items(), key=lambda x: x[1], reverse = True)

    return sorted_emissions

def _clean_text(training_corpus):
    """ Function used for cleaning text from training data that follows the format of the Brown corpus. Closed
        category words and punctuation are not removed to be able to ensure that training sentences are
        syntactically sound"""

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

        #NEED TO REMOVE SPECIAL CHARACTERS NOT CLASSIFIED BY BROWN CORPUS POS GENERATOR!!!

        #Remove extra spaces between words
        lines = [re.sub(' +', ' ', line) for line in lines]

        f.close()

        return lines

def create_model_file(training_corpus):
    """ Function for creating trigram hidden Markov model using unigrams, bigrams, trigrams,
        word tag sequences, and their counts. The model is written onto a file in the library"""

    input_text = _clean_text(training_corpus)

    tag_size = 0
    tag_total = 0

    with open('lib/model_file.txt', 'w+') as f:
        #clear file if contents already exist
        f.truncate(0)

        f.write('@unigrams@' + '\n')
        for key, value in _pos_sequences(input_text, 1):
            tag_size += 1
            tag_total += value
            string = key + '\t' + str(value)
            f.write(string + '\n')

        f.write('@bigrams@' + '\n')
        for key, value in _pos_sequences(input_text, 2):
            string = key + '\t' + str(value)
            f.write(string + '\n')

        f.write('@trigrams@' + '\n')
        for key, value in _pos_sequences(input_text, 3):
            string = key + '\t' + str(value)
            f.write(string + '\n')

        f.write('@emissions@' + '\n')
        for key, value in _emissions(input_text):
            string = key + '\t' + str(value)
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

    create_model_file('lib/train_corpus.txt')

##########################################################################################################
