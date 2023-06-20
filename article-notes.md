# Summary and Analysis of *"Gov2Vec: Learning Distributed Representations of Institutions and Their Legal Text"*

The link to the article can be found [here](https://arxiv.org/abs/1609.06616)

## Introduction

> "Methods have been developed to efficiently obtain representations of words in $\textbf{R}^d$ that capture sublte semantics across the dimenstions of the vectors ([Collobert and Weston, 2008](http://machinelearning.org/archive/icml2008/papers/391.pdf))"

Object embeddings (e.g. word embeddings) are a way of transforming a one-hot vector representing a single object from a set of objects into a presumably lower-dimensional space.

Once created, these embeddings can be used with vector arithmetic that encodes meaning. ([Mikolov et al. 2013a](https://aclanthology.org/N13-1090/))

## Methods

> "A common method for learning vector representations of words is to use a neural network to predict a target word with the mena of its context words' vectors, obtain the gradient with back-propagation of the prediction errors, and update vectors in the direction of higher probability of observing the correct target word ([Bengio et al. 2003](https://www.semanticscholar.org/paper/A-Neural-Probabilistic-Language-Model-Bengio-Ducharme/6c2b28f9354f667cd5bd07afc0471d8334430da7); [Mikolov et al. 2013b](https://papers.nips.cc/paper_files/paper/2013/hash/9aa42b31882ec039965f3c4923ce901b-Abstract.html))."



>"[Le and Mikolov (2014)](https://arxiv.org/abs/1405.4053) extend this word2vec method to learn representations of documents. For predictions of target words, a vector unique to the document is concatenated with context word vectors and subsequently updated. Similarly, we embed institutions and their words into a shared vector space by averaging a vector unique to an institution with context word vectors when predicting that institution’s words and, with back-propagation and stochastic gradient descent, update representations for institutions and the words (which are shared across all institutions). (Note: We use a binary Huffman tree ([Mikolov et al. 2013b](https://papers.nips.cc/paper_files/paper/2013/hash/9aa42b31882ec039965f3c4923ce901b-Abstract.html)) for efficient hierarchical softmax prediction of words, and conduct 25 epochs while linearly decreasing the learning rate from 0.025 to 0.001.)"

> "During training, we alternate between updating GovVecs based on their use in the prediction of words in their policy corpus and their use in the prediction of other word sources located nearby in time."

>"After training, we extract (*M* + *S*) × *d*<sub>*j*</sub> × *J* parameters, where *M* is the number of unique words, *S* is the number of word sources, and *d*<sub>*j*</sub> the vector dimensionality, which varies across the *J* models (we set *J* = 20)."

The extract the *M* × *d*<sub>*j*</sub> parameters for the word embeddings, *S* × *d*<sub>*j*</sub> parameters for the institution embeddings, and *J* models of each embedding.

>"We then investigate the most cosine similar words to particular vector combinations, $\arg\max_{v* \in V_{1:N}} \cos(v*,\frac{1}{W}\sum^W_{i=1}w_i \times s_i)$, where $\cos(a,b)=\frac{\vec{a} \cdot \vec{b}}{\|\vec{a}\|\|\vec{b}\|}$, $w_i$ is one of $W$ WordVecs or GovVecs of interest, $V_{1:N}$ are the $N$ most frequent words in the vocabulary of $M$ words ($N < M$ to exclude rare words during analysis) excluding the $W$ query words, $s_i$ is 1 or -1 for whether we're positively or negatively weighting $w_i$. We repeat similar queries over all $J$ models, retain words with $> C$ cosine similarity, and rank the word results based on their frequency and mean cosine similarity across the ensemble. We also measure the similarity of WordVec combinations to each GovVec and the similarities between GovVecs to validate that the process learns useful embeddings that capture expected relationships"

The "particular vector combinations" are created using vector arithmetic on a query phase. That is, a query $W$ of a set of words is combined, $\frac{1}{W}\sum^W_{i=1}w_i \times s_i$, to create a single query vector.

Once this query vector is created, of the most frequent words in the corpus, the one with the maximum cosine similarity is found. This is then done for all $J$ models to create a set of words that are cosine similar to the query vector. Finally, of this set, a threshold cosine similarity $C$ is set.

This query and similarity pattern can be done between two word vectors, word and institution vectors, or two institution vectors.

## Data

Unique corpus of:
* 59 years of all U.S. Supreme Court opinions (1937-1975, 1991-2010)
    * We downloaded Supreme Court Decisions issued 1937–1975 (Vol. 300-422) from the [GPO](https://www.gpo.gov/fdsys/bulkdata/SCD/1937),
    * and the PDFs of Decisions issued 1991– 2010 (Vol. 502-561) from the [Supreme Court](https://www.supremecourt.gov/opinions/USReports.aspx).
* 227 years of all U.S. Presidential Memorandum, Determinations, and Proclamations, and Executive Orders (1789-2015)
    * We scraped all Presidential Memorandum (1,465),
    * Determinations (801),
    * Executive Orders (5,634),
    * and Proclamations (7,544) from the [American Presidency Project website](https://www.presidency.ucsb.edu/).
* 42 years of official summaries of all bills introduced in the U.S. Congress (1973-2014)
    * The Sunlight Foundation downloaded [official bill summaries](https://github.com/unitedstates/congress/wiki) from the U.S. Government Publishing Office (GPO), which we downloaded.