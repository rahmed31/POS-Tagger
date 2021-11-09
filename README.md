# POS-Tagger
This repository details the creation of a Part-of-Speech tagger using Trigram Hidden Markov Models to predict word tags in a word sequence.

# What is Part-of-Speech Tagging?

In corpus linguistics, part-of-speech tagging (POS tagging, PoS tagging, or POST), also known as "grammatical tagging," is the process of marking up a word in a text (corpus) as corresponding to a particular part of speech, based on both its definition and its context. Once performed by hand, POS tagging is now done in the context of computational linguistics, using algorithms which associate discrete terms, as well as "hidden" parts of speech, by a set of descriptive tags. POS-tagging algorithms fall into two distinctive categories: rule-based and stochastic. Because applying a rule-based model to predict tags in a sequence is cumbersome and restricted to a computational linguist's understanding of allowable sentence construction in the context of language productivity, I'll instead be taking a stochastic approach to assigning POS tags to words in a sequence through the use of Trigram Hidden Markov Models. 

# What are Trigram Hidden Markov Models (HMMs)?

The hidden Markov model, or HMM for short, is a probabilistic sequence model that assigns a label to each unit in a sequence of observations (i.e, input sentences). The model computes a probability distribution over possible sequences of POS labels (using a training corpus) and then chooses the best label sequence that maximizes the probability of generating the observed sequence. The HMM is widely used in natural language processing since language consists of sequences at many levels such as sentences, phrases, words, or even characters. The HMM can be enhanced to incorporate not only unobservable parts-of-speech, but also observable components (i.e., the actual order of words in a sequence) through the use of a probability distribution over the set of trigrams in the given corpus. This allows our model to distinguish between homophones, or words that share the same spelling or pronunciation, but differ in meaning and parts-of-speech (i.e., "rose" as in "rose bush" (NN) and "rose" (VBD) as in the past tense of "rise").
