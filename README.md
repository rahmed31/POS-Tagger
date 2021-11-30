# POS-Tagger
This repository details the creation of a Part-of-Speech tagger using a trigram hidden Markov model and the Viterbi algorithm to predict word tags in a word sequence.

# What is Part-of-Speech Tagging?

In corpus linguistics, part-of-speech tagging (POS tagging, PoS tagging, or POST), also known as "grammatical tagging," is the process of marking up words and punctuations in a text (corpus) as corresponding to particular parts of speech, based on both their definition and their context. Once performed by hand, POS tagging is now done through the use of algorithms which associate discrete terms, as well as "hidden" parts of speech, by a set of descriptive tags. This application merely scratches the surface of computational linguistics. POS-tagging algorithms fall into two distinctive categories: rule-based and stochastic. Because applying a rule-based model to predict tags in a sequence is cumbersome and restricted to a linguist's understanding of allowable sentences in the context of language productivity, a stochastic approach is instead taken to assign POS tags to words in a sequence through the use of a trigram hidden Markov model and the Viterbi algorithm. 

# What are Trigram Hidden Markov Models (HMMs)?

The hidden Markov model, or HMM for short, is a probabilistic sequence model that assigns a label to each unit in a sequence of observations (i.e, input sentences). The model computes a probability distribution over possible sequences of POS labels (from a training corpus) and then chooses the best label sequence that maximizes the probability of generating the observed sequence. The HMM is widely used in natural language processing since language consists of sequences at many levels such as sentences, phrases, words, or even characters. The HMM can be enhanced to incorporate not only unobservable parts-of-speech, but also observable components (i.e., the actual order of words in a sequence) through the use of a probability distribution over the set of trigrams in the given corpus. This allows our model to distinguish between the likes of homophones, or words that share the same spelling or pronunciation, but differ in meaning and parts-of-speech (i.e., "rose" as in "rose bush" (NN) and "rose" (VBD) as in the past tense of "rise").

# Approximating POS tags using Trigram HMMs:

**Decoding** is the task of determining which sequence of variables is the underlying source of some sequence of observations.



