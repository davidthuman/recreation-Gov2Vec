# Paper Summaries

## Efficient Estimation of Word Representations in Vector Space

Published on September 7th, 2013, [this](https://arxiv.org/abs/1301.3781) paper propose the then novel model architectures for computing continuous vector representations of words, being the Continuous Bag-of-Words Model (CBOW) and the Continuous Skip-gram Model (Skip-gram).

## Distributed Representations of Words and Phrases and their Compositionality

Published on October 16th, 2013, [this](https://arxiv.org/abs/1310.4546) paper gives additional improvements to both the quality of the vectors and the training speed when training with a Skip-gram model.

It give the Hierarchical Softmax, "a computationally efficient approximation to the full softamx".

And Noise Contrastive Estimation (NCE)

## Distributed Representations of Sentences and Documents

Published on May 22nd, 2014, [this](https://arxiv.org/abs/1405.4053) paper proposes a method for learning Paragraph vectors, by using one-hot paragraph vectors to learn paragraph embeddings to predict words within that paragraph.

The Distributed Memory Model of Paragraph Vectors (PV-DM) is a CBOW Model with the addition of a one-hot paragraph vector.

The Distributed Bag of Words version of Paragraph Vector (PV-DBOW) is like a Skip-gram with the one-hot paragraph vector as the input to predict a sequence of words.

## Document Embedding with Paragraph Vectors

Published on July 29th, 2015, [this](https://arxiv.org/abs/1507.07998) paper takes a deeper analysis into Paragraph Vectors.

## Paragraph Vectors

[This](https://github.com/inejc/paragraph-vectors/tree/master) GitHub repository implements both the PV-DM and PV-DBOW models. The implementation also uses Negative Sampling objective for model loss.