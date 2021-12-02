#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------------------
#The purpose of this file is to calculate the accuracy of the POS tagging model
#
#Copyright (C) 2021, released under MIT License
#Author: Raihan Ahmed, Chicago, IL
#email: rahmed10@neiu.edu
#-------------------------------------------------------------------------------------------

from train_model import clean_text

output_path = 'data/model_data/'

def calculate_accuracy(test_tags, model_tags):
    """ Function to calculate the accuracy of the Viterbi algorithm by comparing the output of the POS tagger to the actual tags
        provided in the test set. """

    num_correct = 0
    total = 0

    for test_list, model_list in zip(test_tags, model_tags):
        for test_pos, model_pos in zip(test_list, model_list):
            if test_pos == model_pos:
                num_correct += 1

            total += 1

    return round(num_correct/float(total), 3) * 100

if __name__ == '__main__':

    tagged_sentences = pickle.load(open(output_path + "tagged.pickle", "rb"))

    test_sentences, test_tags = clean_text('data/test_corpus.txt')

    model_tags = [[wordtag.rsplit('/', 1)[-1] for wordtag in line.strip().split(" ")] for line in tagged_sentences]

    print("The accuracy of the POS model is:" + calculate_accuracy(test_tags, model_tags) + "%")
