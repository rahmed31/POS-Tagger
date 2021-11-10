#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
__author__ = "Raihan Ahmed"

import sys
import re
import os

def ngram(sentences, n):
    """ Function to find ngrams in a given corpus, where n can equal any number and returns a dictionary
        with the corpus' respective ngrams and their frequencies. Runtime complexity: O(k*m), where k = the number
        of sentences and m = the len(sentence) minus n """

    dictionary = {}

    for sentence in sentences:
        #add n - 1 star symbols to the beginning of the list to account for ngrams of different size
        sentence = ((n - 1) * ['*']) + sentence.split(" ")

        for i in range(len(sentence) - n + 1):
            #joining tokens to create a single string
            token = ' '.join(sentence[i:i+n])

            if token in dictionary:
                dictionary[token] += 1
            else:
                dictionary.update({token : 1})

    #sort dictionary from greatest count to lowest count
    sorted_dictionary = sorted(dictionary.items(), key=lambda x: x[1], reverse = True)

    return sorted_dictionary

def write_to_file(sorted_unigrams, sorted_bigrams, sorted_trigrams, output_file):
    """ Function used for writing unigram, bigram, and trigram contents into a model file"""

    vocab_size = len(sorted_unigrams)
    total_words = 0

    with open(output_file, 'w+') as f:
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

        f.write("@vocab_size@\t" + str(vocab_size) + '\n')
        f.write("@total@\t" + str(total_words))

        f.close()

def clean_text(input_file):
    """ Function used for cleaning text. Auxiliary and closed-category words are not removed to be able to
        to ensure that training sentences are syntactically sound"""

    try:
        #read file in binary
        with open(input_file, 'rb') as f:
            #read all input from file at once
            lines = f.read().decode()

    except IOError:
        print("Error: The input file does not appear to exist! Operation terminated.")
    else:
        #Replace commas with empty string and apostrophes with space, lowercase the entire corpus, replace common
        #end-of-sentence punctuation with "STOP" and remove empty lines
        lines = [line.replace(",", "").replace("'", "").strip().lower() + " STOP" for line in re.split('[.!?:;] ', lines) if len(line.strip())  != 0]

        #Remove any special characters that are not alphanumeric or space
        lines = [re.sub('[^A-Za-z0-9 ]+', '', line) for line in lines]

        #Remove extra spaces between words
        lines = [re.sub(' +', ' ', line) for line in lines]

        f.close()

        return lines

#Main driver used for debugging
if __name__ == '__main__':

    #read input and output files
    inputs = sys.argv[1:]

    #extract necessary lines from input file
    sentences = clean_text(input_file)

    #ngrams dictionary is written to text file
    write_to_file(ngram(sentences, 1), ngram(sentences, 2), ngram(sentences, 3), output_file)

    print("======= Model file successfully created! =======")
