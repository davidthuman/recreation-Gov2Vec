# Libray Imports

import torch
import torch.nn as nn
import math

# Define architecture functions
def context_avg(x): return torch.mean(x, dim=1)
def context_sum(x): return torch.sum(x, dim=1)
def context_cat(x): return torch.reshape(x, (x.size(dim=0), x.size(dim=1) * x.size(dim=2)))
def combine_avg(x, y): return torch.mean(torch.stack((x, y), dim=2), dim=2)
def combine_sum(x, y): return x + y
def combine_cat(x, y): return torch.cat((x, y), dim=1)

class Gov2Vec_Model(nn.Module):
    """ Gov2Vec Model
    """
    def __init__(self, arch_context, arch_gov, vocab_size, gov_size, embed_dim, window_size, loss_weights):
        """ Initializes Gov2Vec Model
        
        :param arch_context: combination architecture for context
        :type arch_context: str
        :param arch_gov: combination architecture for government
        :type arch_gov: str
        :param vocab_size: size of the vocabulary
        :type vocab_size: int
        :param gov_size: size of institutions
        :type gov_size: int
        :param embed_dim: dimension of embedding
        :type embed_dim: int
        :param window_size: context window length from both sides of target
        :type window_size: int
        """

        super(Gov2Vec_Model, self).__init__()
        # Weights should be given to CrossEntropyLoss that incorporate the frequency of words
        # in the dataset. Weights for the minority classes (words) should be higher.
        self.criterion = nn.NLLLoss()
        self.logsoftmax = nn.LogSoftmax(dim=1)

        self.vocab_size = vocab_size
        self.gov_size = gov_size
        self.embed_dim = embed_dim

        self.word_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim
        )

        match arch_context:
            case 'AVG':  # Need optionality for averaging gov and context in a single mean
                self.combine_context = context_avg
                context_size = embed_dim
            case 'SUM':
                self.combine_context = context_sum
                context_size = embed_dim
            case 'CONCAT':
                if arch_gov in ['AVG', 'SUM']:
                    raise ValueError("""An invalid option for `--arch-gov` was supplied,
                                     options for when 'CONCAT' is supplied to `--arch-context` 
                                     are only 'CONCAT'""")
                self.combine_context = context_cat
                context_size = 2 * window_size * embed_dim
            case _:
                raise ValueError("""An invalid option for `--arch-context` was supplied,
                                options are ['AVG', 'SUM', or 'CONCAT']""")

        match arch_gov:
            case 'AVG':
                self.combine = combine_avg
                combine_size = context_size
            case 'SUM':
                self.combine = combine_sum
                combine_size = context_size
            case 'CONCAT':
                self.combine = combine_cat
                combine_size = embed_dim + context_size
            case _:
                raise ValueError("""An invalid option for `--arch-gov` was supplied,
                                options are ['AVG', 'SUM', or 'CONCAT']""")

        self.gov_embedding = nn.Embedding(
            num_embeddings=gov_size,
            embedding_dim=embed_dim
        )
        self.linear = nn.Linear(
            in_features=combine_size,
            out_features=vocab_size
        )

        self.init_weights()

    def init_weights(self):
        # https://stackoverflow.com/questions/49433936/how-do-i-initialize-weights-in-pytorch
        # Need to init weights for both 
        y_gov = 1 / math.sqrt(self.gov_size)
        nn.init.uniform_(self.gov_embedding.weight, -y_gov, y_gov)
        y_word = 1 / math.sqrt(self.vocab_size)
        nn.init.uniform_(self.word_embedding.weight, -y_word, y_word)
        nn.init.zeros_(self.linear.bias)
        nn.init.uniform_(self.linear.weight, -0.1, 0.1)

    def forward(self, context, gov):
        """ Forward pass to model of target word context and government

        :param context: left and right context words
        :type context: torch.Tensor
        :param gov: government institution
        :type gov: torch.Tensor
        :returns: model's guess of the target word
        :rtype: torch.Tensor
        """
        context_embedding = self.word_embedding(context)
        gov_embedding = self.gov_embedding(gov)

        context_embedding = self.combine_context(context_embedding)
        combined = self.combine(gov_embedding, context_embedding)
        out = self.linear(combined)
        likely = self.logsoftmax(out)

        return likely