### File completion list (section will be removed):
**In the order that they're used:**
1. `train_model.py`: :white_check_mark:
2. `emission_probs.py`: :white_check_mark:
3. `transmission_probs.py`: :white_check_mark:
4. `viterbi.py`: :x:
5. `accuracy.py`: :white_check_mark:

# Brown Corpus Part-of-Speech Tagger
This repository details the creation of a Part-of-Speech tagger using a trigram hidden Markov model (HMM) and the Viterbi algorithm to predict word tags in a word sequence. It is trained and evaluated on a real-world text called the Brown corpus, which contains approximately 1 million words from 500 texts across 15 different genres. The Brown corpus was compiled by Henry Kučera and W. Nelson Francis at Brown University in Rhode Island and was published in the United States in 1961. For more information regarding the history of the Brown corpus, [click here](https://en.wikipedia.org/wiki/Brown_Corpus).

To view a subset of the Brown Corpus and how it is annotated, [click here](http://fid.cl/courses/nlp/hw/ca_train.txt).

To view the original manual and tagset used in the Brown Corpus, [click here](http://korpus.uib.no/icame/manuals/BROWN/INDEX.HTM).

-------------------------------------------------------------------------------------------------------------------------------------------------------------

To view the results of the POS Tagger, [click here](#results).

For a basic rundown of the repository, [click here](#rundown-of-respository).

To skip to the section on how to run the POS Tagger yourself, [click here](#how-to-test-the-pos-tagger-yourself).

# What is Part-of-Speech (POS) Tagging?

In corpus linguistics, part-of-speech tagging (POS tagging, PoS tagging, or POST), also known as "grammatical tagging," is the process of marking up words and punctuations in a text (corpus) as corresponding to particular parts of speech, based on both their definition and their context. Once performed by hand, POS tagging is now done through the use of algorithms which associate discrete terms, as well as "hidden" parts of speech, by a set of descriptive tags. This application merely scratches the surface of computational linguistics. POS-tagging algorithms fall into two distinctive categories: rule-based and stochastic. Because applying a rule-based model to predict tags in a sequence is cumbersome and restricted to a linguist's understanding of allowable sentences in the context of language productivity, a stochastic approach is instead taken to assign POS tags to words in a sequence through the use of a trigram hidden Markov model and the Viterbi algorithm. 

# What are Trigram Hidden Markov Models (HMMs)?

The hidden Markov model, or HMM for short, is a probabilistic sequence model that assigns a label to each unit in a sequence of observations (i.e, input sentences). The model computes a probability distribution over possible sequences of POS labels (from a training corpus) and then chooses the best label sequence that maximizes the probability of generating the observed sequence. The HMM is widely used in natural language processing since language consists of sequences at many levels such as sentences, phrases, words, or even characters. The HMM can be enhanced to incorporate not only unobservable parts-of-speech, but also observable components (i.e., the actual order of words in a sequence) through the use of a probability distribution over the set of trigrams in the given corpus. This allows our model to distinguish between the likes of homophones, or words that share the same spelling or pronunciation, but differ in meaning and parts-of-speech (i.e., "rose" as in "rose bush" (NN) and "rose" (VBD) as in the past tense of "rise").

# Approximating POS tags using Trigram HMMs:

**Decoding** is the task of determining which sequence of variables is the underlying source of some sequence of observations. Mathematically, we would like to find the most probable sequence of hidden states <code>*Q = q<sub>1</sub>,q<sub>2</sub>,q<sub>3</sub>,...,q<sub>N</sub>*</code> given as input to `HMM λ = (A,B)` and a sequence of observations <code>*O = o<sub>1</sub>,o<sub>2</sub>,o<sub>3</sub>,...,o<sub>N</sub>*</code> where: *`A`* is a transition probability matrix with each element *a<sub>ij</sub>* representing the probability of moving from a hidden state *q<sub>i</sub>* to another state *q<sub>j</sub>* such that:  

<p align="center">
<img width="70" height="50" src=https://latex.codecogs.com/gif.latex?%5Csum_%7Bj%3D1%7D%5E%7Bn%7Da%7B_%7Bij%7D%7D%20%3D%201>
</p>

for ∀<sub>i</sub>, and *`B`* is a matrix of emission probabilities with each element representing the probability of an observation state *o<sub>i</sub>* being generated from a hidden state *q<sub>i</sub>*. In POS tagging, each hidden state corresponds to a single tag, and each observation state corresponds to a word in a given sentence. For example, the task of the decoder may be to find the best hidden tag sequence `DT NNS VB` that maximizes the probability of the observed sequence of words `The boys eat`.

The task of decoding is ultimately defined as Eq. 1:

<p align="center">
<img width="385" height="45" src=https://latex.codecogs.com/gif.latex?q_%7B1%7D%5E%7B%5Cnot%7Bn%7D%7D%3Dargmax_%7Bq_%7B1%7D%5E%7Bn%7D%7DP%28q_%7B1%7D%5E%7Bn%7D%7Co_%7B1%7D%5E%7Bn%7D%29%3Dargmax_%7Bq_%7B1%7D%5E%7Bn%7D%7D%5Cfrac%7BP%28o_%7B1%7D%5E%7Bn%7D%7Cq_%7B1%7D%5E%7Bn%7D%29P%28q_%7B1%7D%5E%7Bn%7D%29%7D%7BP%28o_%7B1%7D%5E%7Bn%7D%29%7D>
</p>

where the second equality is computed using Bayes' rule. Moreover, the denominator of the second equality in Eq. 1 can be dropped since it does not depend on *q*. This gives us Eq. 2:

<p align="center">
<img width="370" height="23" src=https://latex.codecogs.com/gif.latex?q_%7B1%7D%5E%7B%5Cnot%7Bn%7D%7D%3Dargmax_%7Bq_%7B1%7D%5E%7Bn%7D%7DP%28o_%7B1%7D%5E%7Bn%7D%7Cq_%7B1%7D%5E%7Bn%7D%29P%28q_%7B1%7D%5E%7Bn%7D%29%3Dargmax_%7Bq_%7B1%7D%5E%7Bn%7D%7DP%28o_%7B1%7D%5E%7Bn%7D%2Cq_%7B1%7D%5E%7Bn%7D%29>
</p>

The trigram HMM tagger makes two assumptions to simplify the computation of the first equality in Eq. 2. The first is that the **emission** probability of a word appearing depends only on its own tag and is independent of neighboring words and tags, which gives us Eq. 3:

<p align="center">
<img width="190" height="50" src=https://latex.codecogs.com/gif.latex?P%28o_%7B1%7D%5E%7Bn%7D%7Cq_%7B1%7D%5E%7Bn%7D%29%3D%5Cprod_%7Bi%3D1%7D%5E%7Bn%7DP%28o_%7Bi%7D%7Cq_%7Bi%7D%29>
</p>
 
The second is a Markov assumption that the **transition** probability of a tag is dependent only on the previous two tags rather than the entire tag sequence, which gives us Eq. 4:

<p align="center">
<img width="205" height="55" src=https://latex.codecogs.com/gif.latex?P%28q_%7B1%7D%5E%7Bn%7D%29%5Capprox%20%5Cprod_%7Bi%3D1%7D%5E%7Bn&plus;1%7DP%28q_%7Bi%7D%7Cq_%7Bi-1%7D%2Cq_%7Bi-2%7D%29>
</p>

where *q<sub>-1</sub> = q<sub>-2</sub>* = * is the special start symbol appended to the beginning of every tag sequence and *q<sub>n+1</sub>* = *STOP* is the unique stop symbol marked at the end of every tag sequence.

In many cases, we have a labeled corpus of sentences paired with the correct POS tag sequences `The/DT boys/NNS eat/VB` such as in the Brown corpus, so the problem of POS tagging is that of the supervised learning portion, where we can easily calculate the maximum likelihood estimate of a transition probability *P*(*q<sub>i</sub>* | *q<sub>i-1</sub>, q<sub>i-2</sub>*) by counting how often we see the third tag *q<sub>i</sub>* followed by its previous two tags *q<sub>i-1</sub>* and *q<sub>i-2</sub>* divided by the number of occurrences of the two tags *q<sub>i-1</sub>* and *q<sub>i-2</sub>* (Eq. 5):

<p align="center">
<img width="250" height="47" src=https://latex.codecogs.com/gif.latex?P%28q_%7Bi%7D%7Cq_%7Bi-1%7D%2Cq_%7Bi-2%7D%29%20%3D%20%5Cfrac%7BC%28q_%7Bi-2%7D%2Cq_%7Bi-1%7D%2Cq_%7Bi%7D%29%7D%7BC%28q_%7Bi-2%7D%2Cq_%7Bi-1%7D%29%7D>
</p>

Likewise, we compute an emission probability *P*(*o<sub>i</sub>* | *q<sub>i</sub>*) using Eq. 6 as follows:

<p align="center">
<img width="145" height="47" src=https://latex.codecogs.com/gif.latex?P%28o_%7Bi%7D%7Cq_%7Bi%7D%29%20%3D%20%5Cfrac%7BC%28q_%7Bi%7D%2Co_%7Bi%7D%29%7D%7BC%28q_%7Bi%7D%29%7D>
</p>

# The Viterbi Algorithm

Recall that the decoding task is to find:

<p align="center">
<img width="250" height="30" src=https://latex.codecogs.com/gif.latex?q_%7B1%7D%5E%7B%5Cnot%7Bn&plus;1%7D%7D%3Dargmax_%7Bq_%7B1%7D%5E%7Bn&plus;1%7D%7DP%28o_%7B1%7D%5E%7Bn%7D%2Cq_%7B1%7D%5E%7Bn&plus;1%7D%29>
</p>

where the argmax is taken over all sequences *q*<sub>1</sub>*<sup>n</sup>* such that *q<sub>i</sub>* ∈ *S* for *i* = 1,...,*n* and *S* is the set of all tags. We further assume that *P*(*o*<sub>1</sub>*<sup>n</sup>*,*q*<sub>1</sub>*<sup>n</sup>*) takes the form:

<p align="center">
<img width="310" height="50" src=https://latex.codecogs.com/gif.latex?P%28o_%7B1%7D%5E%7Bn%7D%2Cq_%7B1%7D%5E%7Bn&plus;1%7D%29%20%3D%20%5Cprod_%7Bi%3D1%7D%5E%7Bn&plus;1%7DP%28q_%7Bi%7D%20%7C%20q_%7Bt-1%7D%2C%20q_%7Bt-2%7D%29%5Cprod_%7Bi%3D1%7D%5E%7Bn%7DP%28o_%7Bi%7D%7C%20q_%7Bi%7D%29>
</p>

assuming that *q*<sub>-1</sub> and *q*<sub>-2</sub> = * and *q*<sub>n+1</sub> = *STOP*. Because the argmax is taken over all different tag sequences, brute force search by computing the likelihood of the observation sequence given each possible hidden state sequence is very inefficient, as it is completed in O(|*S*|<sup>3</sup>) complexity. Instead, the **Viterbi algorithm**, a kind of dynamic programming algorithm, is used to make the search computationally more efficient.

Define *n* to be the length of the input sentence and *S<sub>k</sub>* for *k* = -1, 0,...,*n* to be the set of possible tags at position *k* such that *S*<sub>-1</sub> = *S*<sub>0</sub> = * and *S<sub>k</sub>* = *Sk* ∈ 1,...,*n*. Define

<p align="center">
<img width="280" height="50" src=https://latex.codecogs.com/gif.latex?r%28q_%7B-1%7D%5E%7Bk%7D%29%20%3D%20%5Cprod_%7Bi%3D1%7D%5E%7Bn&plus;1%7DP%28q_%7Bi%7D%20%7C%20q_%7Bt-1%7D%2C%20q_%7Bt-2%7D%29%5Cprod_%7Bi%3D1%7D%5E%7Bn%7DP%28o_%7Bi%7D%7C%20q_%7Bi%7D%29>
</p>

and a dynamic programming table, or a cell, to be:

<p align="center">
<img width="280" height="28" src=https://latex.codecogs.com/gif.latex?%5Cpi%28k%2C%20u%2C%20v%29%20%3D%20max_%7Bq_%7B-1%7D%5E%7Bk%7D%3Aq_%7Bk-1%7D%20%3Du%2Cq_%7Bk%7D%20%3Dv%7Dr%28q_%7B-1%7D%5E%7Bk%7D%29>
</p>

which is the maximum probability of a tag sequence ending in tags *u*, *v* at position *k*. The Viterbi algorithm fills each cell recursively such that the most probable of the extensions of the paths that lead to the current cell at time *k* given that we had already computed the probability of being in every state at time *k*-1. The algorithm essentially works by initializing the first cell as:

<p align="center">
<img width="100" height="20" src=https://latex.codecogs.com/gif.latex?%5Cpi%280%2C*%2C*%29%3D%201>
</p>

and for any *k* ∈ 1,...,*n*, for any *u* ∈ *S<sub>k-1</sub>* and *v* ∈ *S<sub>k</sub>*, recursively compute:

<p align="center">
<img width="480" height="22" src=https://latex.codecogs.com/gif.latex?%5Cpi%28k%2C%20u%2C%20v%29%20%3D%20max_%7Bw%5Cin%7BS_%7Bk-2%7D%7D%7D%28%5Cpi%28k-1%2Cw%2Cu%29%5Ccdot%20q%28v%7C%20w%2Cu%29%5Ccdot%20P%28o_%7Bk%7D%7C%20v%29%29>
</p>

and ultimately return:

<p align="center">
<img width="340" height="22" src=https://latex.codecogs.com/gif.latex?max_%7Bw%5Cin%7BS_%7Bn-1%7D%7D%2Cv%5Cin%7BS_%7Bn%7D%7D%7D%28%5Cpi%28n%2Cu%2Cv%29%5Ccdot%20q%28STOP%7C%20u%2Cv%29%29>
</p>

The last component of the Viterbi algorithm is **backpointers**. The goal of the decoder is to not only produce a probability of the most probable tag sequence but also the resulting tag sequence itself. The best state sequence is computed by keeping track of the path of hidden state that led to each state and backtracing the best path in reverse from the end to the start.

# Enhancing the POS Tagger
Though utilizing a hidden Markov model in conjunction with the Viterbi algorithm can produce tagging results with approximately 50-70% accuracy on a test corpus, this is still well below the human agreement upper bound of 97% for POS tagging. To be able to approach a higher accuracy rate for POS tagging, two additional features are utitlized to enhance my POS tagging model. These features are detailed below: 

### Deleted Interpolation

Normally, transition probabilities are calculated using Equation 5 above. However, these counts may result in a returned value of zero using a training corpus which erroneously predicts that a given tag sequence will never occur at all. A common, effective solution to this division by zero error is to estimate a trigram transition probability by aggregating weaker, yet more robust estimators such as bigram and unigram probabilities. For instance, assume we have never seen the tag sequence `DT NNS VB` in a training corpus, so the trigram transition probability *P*(*VB* ∣ *DT*,*NNS*) = 0 but it may still be possible to compute the bigram transition probability *P*(*VB* | *NNS*) as well as the unigram probability *P*(*VB*).

More generally, the maximum likelihood estimates of the following transition probabilities can be computed using counts from a training corpus and subsequenty setting them to zero if the denominator happens to be zero:

<p align="center">
<img width="245" height="45" src=https://latex.codecogs.com/gif.latex?P%5Chat%28q_%7Bi%7D%20%7C%20q_%7Bi-1%7D%2C%20q_%7Bi-2%7D%29%20%3D%20%5Cfrac%7BC%28q_%7Bi-2%7D%2C%20q_%7Bi-1%7D%2C%20q_%7Bi%7D%29%7D%7BC%28q_%7Bi-2%7D%2C%20q_%7Bi-1%7D%29%7D>
</p>

<p align="center">
<img width="175" height="40" src=https://latex.codecogs.com/gif.latex?P%5Chat%28q_%7Bi%7D%20%7C%20q_%7Bi-1%7D%29%20%3D%20%5Cfrac%7BC%28q_%7Bi-1%7D%2C%20q_%7Bi%7D%29%7D%7BC%28q_%7Bi-1%7D%29%7D>
</p>

<p align="center">
<img width="100" height="40" src=https://latex.codecogs.com/gif.latex?P%5Chat%28q_%7Bi%7D%29%20%3D%20%5Cfrac%7BC%28q_%7Bi%7D%29%7D%7BC%28N%29%7D>
</p>

where *N* is the total number of tokens, not unique words, in the training corpus. The final trigram probability estimate *P*<sup>~</sup>(*q<sub>i</sub>* | *q<sub>i-1</sub>*, *q<sub>i-2</sub>*) is calculated by a weighted sum of the trigram, bigram, and unigram probability estimates above:

<p align="center">
<img width="440" height="23" src=https://latex.codecogs.com/gif.latex?P%5Ctilde%28q_%7Bi%7D%20%7C%20q_%7Bi-1%7D%2C%20q_%7Bi-2%7D%29%20%3D%20%5Clambda_%7B3%7D%5Ccdot%20P%5Chat%28q_%7Bi%7D%20%7C%20q_%7Bi-1%7D%2C%20q_%7Bi-2%7D%29%20&plus;%20%5Clambda_%7B2%7D%5Ccdot%20P%5Chat%28q_%7Bi%7D%20%7C%20q_%7Bi-1%7D%29%20&plus;%20%5Clambda_%7B1%7D%5Ccdot%20P%5Chat%28q_%7Bi%7D%29>
</p>

under the constraint λ<sub>1</sub> + λ<sub>2</sub> + λ<sub>3</sub> = 1. These values of λs are generally set using the algorithm called deleted interpolation which is conceptually similar to leave-one-out cross-validation `LOOCV` in that each trigram is successively deleted from the training corpus and the λs are chosen to maximize the likelihood of the rest of the corpus. The deletion mechanism thereby helps set the λs so as to not overfit the training corpus and aid in generalization. The λ values are experimentally determined in my repository.

### Morphosyntactic Subcategorization

In linguistics, Hockett's Design Features are a set of features that characterize human language and set it apart from animal communication. Of these features, one of the most important is "productivity," which refers to the idea that language-users can produce and understand an unlimited amount of novel utterances. Also related to productivity is the concept of grammatical patterning, which facilitates the use and comprehension of language. Language is not stagnant, but is constantly changing. Thus, in all human languages, new words and phrases are constantly being coined and added to a dictionary. Updating a dictionary of vocabularies is, however, too cumbersome and takes an unreasonable amount of effort. Because of this, it is important to have a good model for dealing with unknown words found in a test corpus to achieve a high accuracy with a trigram HMM POS tagger.

Utilizing RARE is a simple way to replace every word or token with the special symbol `_RARE_` whose frequency in the training set is less than or equal to 5. Without this process, words like person names and places that do not appear in the training set but are seen in the test set would have their maximum likelihood estimates of P(*q<sub>i</sub>* ∣ *o<sub>i</sub>*) (i.e., the emission probabilities) undefined.

Morphosyntactic subcategorization is a modification of RARE that serves as a better alternative in that every word token whose frequency is less than or equal to 5 in the training set is replaced by further subcategorization based on a set of morphological characteristics (i.e., affixation). For example, we know that words with suffixes like `-ion`, `-ment`, `-ence`, or `-ness`, just to name a few, will be a noun, and that adjectives may possess `un-` and `in-` as prefixes or `-ious` and `-ble` as suffixes. 

# Results

The Trigram HMM POS tagger is trained on a subset of the Brown corpus, which contains nearly 27500 tagged sentences in total. The training set contains approximately 80% of the entire corpus, with the remainder 20% being utilized as the test set. The accuracy of the tagger is measured by comparing the predicted tags in the test set with the true tags of the test set already provided in the corpus. Thus, the percentage of tags that the model gets correct is defined as the accuracy. 

Using a combination of deleted interpolation with morphosyntactic subcategorization, my POS tagger achieves an overall accuracy of: **IN PROGRESS**

# Rundown of Respository 

Each file is **briefly** explained in the order that they were created and used: 

1. `train_model.py` is used to: get two separate lists containing cleaned sentences and POS sequences, respectively, from the training corpus, create dictionaries of POS unigrams, bigrams, and trigrams, and build a sample file of what these dictionaries contain at first glance for the sole purpose of visualization.
2. `emission_probs.py` is used to apply morphosyntactic subcategorization to the sentences list to be able to calculate the emission probabilities for each word/POS pair as well as get a list of "known words" from the sentences lists. 
3. `transmission_probs.py` is used to experimentally calculate the λ values for deleted interpolation, find the interpolated transmission probabilities for each ngram, and ultimtately deduce the overall accuracy of the POS tagger using necessary components retrieved from `train_model.py` and `emission_probs.py`.
4. `viterbi.py` is used to apply the Viterbi algorithm to retrieve the most probabilistic sequence of POS tags for each sentence in the test set. 
5. `accuracy.py` is used to find the accuracy of the Viterbi algorithm by comparing the calculated POS sequences for each test sentence to the actual POS sequences for each test sentence. 

**Each file contains an in-depth description of how they work and the purpose of all functions that are within them.** While looking at each file in order, you will also notice how, and the order in which, I calculate all necessary pieces of data (found in each of my main drivers) to ultimately create my POS tagger. Lastly, you will notice that I modularize all functions in the event that they needed to be imported and reused in different files.

**THIS SECTION WILL CONTINUE TO BE MODIFIED UNTIL REPOSITORY IS FINALIZED**

# How to Test the POS Tagger Yourself

**SECTION INFORMATION WILL BE ADDED WHEN REPOSITORY IS FINALIZED**

# Resources

**TO BE COMPILED**

