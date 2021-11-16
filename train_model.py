#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
__author__ = "Raihan Ahmed"

import sys
import re
import os

def _ngrams(sentences, n):
    """ Function to find ngrams of specified length 'n'. It returns a dictionary
        containing the corpus' ngrams and their frequencies. This function specifically caters to files
        with the format of the Brown corpus. Runtime complexity: O(n^2) """

    ngrams = {}

    for sentence in sentences:
        #add n - 1 star symbols to the beginning of the list to account for ngrams of different size
        word_list = ((n - 1) * ['*'])

        for word in sentence.split(" "):
            word = word.rsplit('/', 1)[0]
            word_list.append(word)

        for i in range(len(word_list) - n + 1):
            #joining tokens in word_list to create a single string
            token = ' '.join(word_list[i:i+n])

            if token in ngrams:
                ngrams[token] += 1
            else:
                ngrams.update({token : 1})

    #sort ngrams dictionary from greatest count to lowest count
    sorted_ngrams = sorted(ngrams.items(), key=lambda x: x[1], reverse = True)

    return sorted_ngrams

def _pos_sequences(sentences, n):
    """ Function to find pos sequences of length n in Brown training corpus. It returns a dictionary
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

def _clean_text(input_file):
    """ Function used for cleaning text from training data that follows the format of the Brown corpus. Closed
        category words and punctuation are not removed to be able to ensure that training sentences are
        syntactically sound"""

    try:
        #read text file
        with open(input_file, 'r') as f:
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

def create_model_file(input_file):
    """ Function for creating trigram hidden Markov model using unigrams, bigrams, trigrams,
        word tag sequences, and their counts. The model is written onto a file in the library"""

    input_text = _clean_text(input_file)

    sorted_unigrams = _ngrams(input_text, 1)
    sorted_tags_1 = _ngrams(input_text, 1)

    sorted_bigrams = _ngrams(input_text, 2)
    sorted_tags_2 = _ngrams(input_text, 2)

    sorted_trigrams = _ngrams(input_text, 3)
    sorted_tags_3 = _ngrams(input_text, 3)

    vocab_size = len(sorted_unigrams)
    tag_size = len(sorted_tags_1)
    total_words = 0

    with open('lib/model_file.txt', 'w+') as f:
        #clear file if contents already exist
        f.truncate(0)

        for key, value in sorted_unigrams:
            total_words += value
            string = key + '\t' + str(value)
            f.write(string + '\n')

        for key, value in sorted_bigrams:
            string = key + '\t' + str(value)
            f.write(string + '\n')

        for key, value in sorted_trigrams:
            string = key + '\t' + str(value)
            f.write(string + '\n')

        for key, value in sorted_tags_1:
            string = key + '\t' + str(value)
            f.write(string + '\n')

        for key, value in sorted_tags_2:
            string = key + '\t' + str(value)
            f.write(string + '\n')

        for key, value in sorted_tags_3:
            string = key + '\t' + str(value)
            f.write(string + '\n')

        f.write("@vocab_size@\t" + str(vocab_size) + '\n')
        #total_words also equals total tags
        f.write("@total_words@\t" + str(total_words) + '\n')
        #tag_size is not necessarily the same as vocab_size (need to double check this)
        f.write("@tag_size@\t" + str(tag_size) + '\n')

        f.close()

    print("--------------------------------")
    print("Model file successfully created!")
    print("--------------------------------")

#Main driver used for debugging
if __name__ == '__main__':

    create_model_file('lib/train_corpus.txt')

##########################################################################################################

# def pos_ngrams(sentences, n):
#     """ Function to find ngrams and pos sequences of length n in Brown training corpus. It returns two dictionaries:
#         one containing the corpus' pos sequences and their frequencies, and the other containing the corpus' ngrams and
#         their frequencies. This function specifically caters to the format of the Brown corpus. Runtime complexity: O(n^2) """
#
#     ngrams = {}
#     tag_sequences = {}
#
#     for sentence in sentences:
#
#         word_list = []
#         tags_list = []
#
#         for word in sentence.split(" "):
#             word, pos = word.rsplit('/', 1)
#             word_list.append(word)
#             tags_list.append(pos)
#
#         for i in range(len(word_list) - n + 1):
#             #joining tokens in word_list to create a single string
#             token = ' '.join(word_list[i:i+n])
#
#             #joining pos tags in tags_list to create a single tag sequence
#             sequence = ' '.join(tags_list[i:i+n])
#
#             if token in ngrams:
#                 ngrams[token] += 1
#             else:
#                 ngrams.update({token : 1})
#
#             if sequence in tag_sequences:
#                 tag_sequences[sequence] += 1
#             else:
#                 tag_sequences.update({sequence : 1})
#
#     #sort ngrams dictionary from greatest count to lowest count
#     sorted_ngrams = sorted(ngrams.items(), key=lambda x: x[1], reverse = True)
#
#     #sort tag_sequences dictionary from greatest count to lowest count
#     sorted_tag_sequences = sorted(tag_sequences.items(), key=lambda x: x[1], reverse = True)
#
#     return sorted_ngrams, sorted_tag_sequences
