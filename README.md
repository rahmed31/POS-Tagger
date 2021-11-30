# POS-Tagger
This repository details the creation of a Part-of-Speech tagger using a trigram hidden Markov model and the Viterbi algorithm to predict word tags in a word sequence.

# What is Part-of-Speech Tagging?

In corpus linguistics, part-of-speech tagging (POS tagging, PoS tagging, or POST), also known as "grammatical tagging," is the process of marking up words and punctuations in a text (corpus) as corresponding to particular parts of speech, based on both their definition and their context. Once performed by hand, POS tagging is now done through the use of algorithms which associate discrete terms, as well as "hidden" parts of speech, by a set of descriptive tags. This application merely scratches the surface of computational linguistics. POS-tagging algorithms fall into two distinctive categories: rule-based and stochastic. Because applying a rule-based model to predict tags in a sequence is cumbersome and restricted to a linguist's understanding of allowable sentences in the context of language productivity, a stochastic approach is instead taken to assign POS tags to words in a sequence through the use of a trigram hidden Markov model and the Viterbi algorithm. 

# What are Trigram Hidden Markov Models (HMMs)?

The hidden Markov model, or HMM for short, is a probabilistic sequence model that assigns a label to each unit in a sequence of observations (i.e, input sentences). The model computes a probability distribution over possible sequences of POS labels (from a training corpus) and then chooses the best label sequence that maximizes the probability of generating the observed sequence. The HMM is widely used in natural language processing since language consists of sequences at many levels such as sentences, phrases, words, or even characters. The HMM can be enhanced to incorporate not only unobservable parts-of-speech, but also observable components (i.e., the actual order of words in a sequence) through the use of a probability distribution over the set of trigrams in the given corpus. This allows our model to distinguish between the likes of homophones, or words that share the same spelling or pronunciation, but differ in meaning and parts-of-speech (i.e., "rose" as in "rose bush" (NN) and "rose" (VBD) as in the past tense of "rise").

# Approximating POS tags using Trigram HMMs:

**Decoding** is the task of determining which sequence of variables is the underlying source of some sequence of observations. Mathematically, we want to find the most probable sequence of hidden states *`Q = q_1,q_2,q_3,...,q_N`* given as input to `HMM λ = (A,B)` and a sequence of observations *`O = o_1,o_2,o_3,...,o_N`* where: *`A`* is a transition probability matrix with each element *`a_ij`* representing the probability of moving from a hidden state *`q_i`* to another state *`q_j`* such that the *`summation from j=1 to n of a_ij equals 1`* for *`∀i`*, and *`B`* is a matrix of emission probabilities with each element representing the probability of an observation state *`o_i`* being generated from a hidden state *`q_i`*. In POS tagging, each hidden state corresponds to a single tag, and each observation state corresponds to a word in a given sentence. For example, the task of the decoder may be to find the best hidden tag sequence `DT NNS VB` that maximizes the probability of the observed sequence of words `The dogs run`.

The task of decoding is given as:

![equation](https://latex.codecogs.com/gif.latex?q_%7B1%7D%5E%7Bn%7D%3Dargmax_%7Bq_%7B1%7D%5E%7Bn%7D%7DP%28q_%7B1%7D%5E%7Bn%7D%7Co_%7B1%7D%5E%7Bn%7D%29%3Dargmax_%7Bq_%7B1%7D%5E%7Bn%7D%7D%5Cfrac%7BP%28o_%7B1%7D%5E%7Bn%7D%7Cq_%7B1%7D%5E%7Bn%7D%29P%28q_%7B1%7D%5E%7Bn%7D%29%7D%7BP%28o_%7B1%7D%5E%7Bn%7D%29%7D)

where the second equality is computed using Bayes' rule. Moreover, the denominator of the second equality in the equation above can be dropped since it does not depend on *q*. This gives us:

![equation](https://latex.codecogs.com/gif.latex?q_%7B1%7D%5E%7Bn%7D%3Dargmax_%7Bq_%7B1%7D%5E%7Bn%7D%7DP%28o_%7B1%7D%5E%7Bn%7D%7Cq_%7B1%7D%5E%7Bn%7D%29P%28q_%7B1%7D%5E%7Bn%7D%29%3Dargmax_%7Bq_%7B1%7D%5E%7Bn%7D%7DP%28o_%7B1%7D%5E%7Bn%7D%2Cq_%7B1%7D%5E%7Bn%7D%29)

