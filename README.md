# POS-Tagger
This repository details the creation of a Part-of-Speech tagger using a trigram hidden Markov model and the Viterbi algorithm to predict word tags in a word sequence.

# What is Part-of-Speech Tagging?

In corpus linguistics, part-of-speech tagging (POS tagging, PoS tagging, or POST), also known as "grammatical tagging," is the process of marking up words and punctuations in a text (corpus) as corresponding to particular parts of speech, based on both their definition and their context. Once performed by hand, POS tagging is now done through the use of algorithms which associate discrete terms, as well as "hidden" parts of speech, by a set of descriptive tags. This application merely scratches the surface of computational linguistics. POS-tagging algorithms fall into two distinctive categories: rule-based and stochastic. Because applying a rule-based model to predict tags in a sequence is cumbersome and restricted to a linguist's understanding of allowable sentences in the context of language productivity, a stochastic approach is instead taken to assign POS tags to words in a sequence through the use of a trigram hidden Markov model and the Viterbi algorithm. 

# What are Trigram Hidden Markov Models (HMMs)?

The hidden Markov model, or HMM for short, is a probabilistic sequence model that assigns a label to each unit in a sequence of observations (i.e, input sentences). The model computes a probability distribution over possible sequences of POS labels (from a training corpus) and then chooses the best label sequence that maximizes the probability of generating the observed sequence. The HMM is widely used in natural language processing since language consists of sequences at many levels such as sentences, phrases, words, or even characters. The HMM can be enhanced to incorporate not only unobservable parts-of-speech, but also observable components (i.e., the actual order of words in a sequence) through the use of a probability distribution over the set of trigrams in the given corpus. This allows our model to distinguish between the likes of homophones, or words that share the same spelling or pronunciation, but differ in meaning and parts-of-speech (i.e., "rose" as in "rose bush" (NN) and "rose" (VBD) as in the past tense of "rise").

# Approximating POS tags using Trigram HMMs:

**Decoding** is the task of determining which sequence of variables is the underlying source of some sequence of observations. Mathematically, we want to find the most probable sequence of hidden states <code>*Q = q<sub>1</sub>,q<sub>2</sub>,q<sub>3</sub>,...,q<sub>N</sub>*</code> given as input to `HMM λ = (A,B)` and a sequence of observations <code>*O = o<sub>1</sub>,o<sub>2</sub>,o<sub>3</sub>,...,o<sub>N</sub>*</code> where: *`A`* is a transition probability matrix with each element *a<sub>ij</sub>* representing the probability of moving from a hidden state *q<sub>i</sub>* to another state *q<sub>j</sub>* such that the summation from j=1 to n of *a<sub>ij</sub>* equals 1 for ∀<sub>i</sub>, and *`B`* is a matrix of emission probabilities with each element representing the probability of an observation state *o<sub>i</sub>* being generated from a hidden state *q<sub>i</sub>*. In POS tagging, each hidden state corresponds to a single tag, and each observation state corresponds to a word in a given sentence. For example, the task of the decoder may be to find the best hidden tag sequence `DT NNS VB` that maximizes the probability of the observed sequence of words `The dogs run`.

The task of decoding is ultimately defined as Eq. 1:

![equation](https://latex.codecogs.com/gif.latex?q_%7B1%7D%5E%7Bn%7D%3Dargmax_%7Bq_%7B1%7D%5E%7Bn%7D%7DP%28q_%7B1%7D%5E%7Bn%7D%7Co_%7B1%7D%5E%7Bn%7D%29%3Dargmax_%7Bq_%7B1%7D%5E%7Bn%7D%7D%5Cfrac%7BP%28o_%7B1%7D%5E%7Bn%7D%7Cq_%7B1%7D%5E%7Bn%7D%29P%28q_%7B1%7D%5E%7Bn%7D%29%7D%7BP%28o_%7B1%7D%5E%7Bn%7D%29%7D)

where the second equality is computed using Bayes' rule. Moreover, the denominator of the second equality in Eq. 1 can be dropped since it does not depend on *q*. This gives us Eq. 2:

![equation](https://latex.codecogs.com/gif.latex?q_%7B1%7D%5E%7Bn%7D%3Dargmax_%7Bq_%7B1%7D%5E%7Bn%7D%7DP%28o_%7B1%7D%5E%7Bn%7D%7Cq_%7B1%7D%5E%7Bn%7D%29P%28q_%7B1%7D%5E%7Bn%7D%29%3Dargmax_%7Bq_%7B1%7D%5E%7Bn%7D%7DP%28o_%7B1%7D%5E%7Bn%7D%2Cq_%7B1%7D%5E%7Bn%7D%29)

The trigram HMM tagger makes two assumptions to simplify the computation of the first equality in Eq. 2. The first is that the **emission** probability of a word appearing depends only on its own tag and is independent of neighboring words and tags:

![equation](https://latex.codecogs.com/gif.latex?P%28o_%7B1%7D%5E%7Bn%7D%7Cq_%7B1%7D%5E%7Bn%7D%29%3D%5Cprod_%7Bi%3D1%7D%5E%7Bn%7DP%28o_%7Bi%7D%7Cq_%7Bi%7D%29)

The second is a Markov assumption that the **transition** probability of a tag is dependent only on the previous two tags rather than the entire tag sequence:

![equation](https://latex.codecogs.com/gif.latex?P%28q_%7B1%7D%5E%7Bn%7D%29%5Capprox%20%5Cprod_%7Bi%3D1%7D%5E%7Bn&plus;1%7DP%28q_%7Bi%7D%7Cq_%7Bi-1%7D%2Cq_%7Bi-2%7D%29)

where *q<sub>-1</sub> = q<sub>-2</sub>* = * is the special start symbol appended to the beginning of every tag sequence and *q<sub>n+1</sub>* = *STOP* is the unique stop symbol marked at the end of every tag sequence.

In many cases, we have a labeled corpus of sentences paired with the correct POS tag sequences `The/DT dogs/NNS run/VB` such as in the Brown corpus, so the problem of POS tagging is that of the supervised learning where we easily calculate the maximum likelihood estimate of a transition probability *P*(*q<sub>i</sub>* | *q<sub>i-1</sub>, q<sub>i-2</sub>*) by counting how often we see the third tag *q<sub>i</sub>* followed by its previous two tags *q<sub>i-1</sub>* and *q<sub>i-2</sub>* divided by the number of occurrences of the two tags *q<sub>i-1</sub>* and *q<sub>i-2</sub>*:

![equation](https://latex.codecogs.com/gif.latex?P%28q_%7Bi%7D%7Cq_%7Bi-1%7D%2Cq_%7Bi-2%7D%29%20%3D%20%5Cfrac%7BC%28q_%7Bi-2%7D%2Cq_%7Bi-1%7D%2Cq_%7Bi%7D%29%7D%7BC%28q_%7Bi-2%7D%2Cq_%7Bi-1%7D%29%7D)

Likewise, we compute an emission probability *P*(*o<sub>i</sub>* | *q<sub>i</sub>*) as follows:

![equation](https://latex.codecogs.com/gif.latex?P%28o_%7Bi%7D%7Cq_%7Bi%7D%29%20%3D%20%5Cfrac%7BC%28q_%7Bi%7D%2Co_%7Bi%7D%29%7D%7BC%28q_%7Bi%7D%29%7D)
 
