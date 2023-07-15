# Hierarchical Softmax

YouTube [video](https://www.youtube.com/watch?v=B95LTf2rVWM) explaining the implementation.

The output of the hierarchical softmax is the probability of a single word. This process can be done a few times if multiple probabilities are wanted, however, if the whole probability distribution is needed, using a H-Softmax gives no speed-up.

Video suggests checking out the paper below

##  A Scalable Hierarchical Distributed Language Model

[This](https://www.cs.toronto.edu/~amnih/papers/hlbl_final.pdf) paper 

[Article](https://papers.nips.cc/paper_files/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf) cited when mentioning of Hierarchical Softmax

[PyTorch Discussion post](https://discuss.pytorch.org/t/feedback-on-manually-implemented-hierarchical-softmax/82478)

[Two Layer Hierarchical Softmax PyTorch](https://github.com/leimao/Two-Layer-Hierarchical-Softmax-PyTorch/tree/master) with associated article [post](https://leimao.github.io/article/Hierarchical-Softmax/).

Another GitHub [implementation](https://github.com/rbturnbull/hierarchicalsoftmax)