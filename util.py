# Data imports
from collections import Counter
from itertools import chain
import torch
from torch.utils.data import Dataset, DataLoader
import math
import random
import os
import tqdm
import sys
import csv
# Query imports
import torch
import tqdm

# Set CSV size to max
maxInt = sys.maxsize
while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

###############################################################################
# Data functions and classes
###############################################################################

UNK = "<UNK>"
PAD = "<PAD>"

class Vocab(object):
    """ Vocabulary, i.e. structure containing language terms.
        This vocabulary can, and should, be abstracted to other
        sets of object. For vocabulary, it is words to tokens. 
        For government institutions, it is institutions to tokens.

        Instance attributes:
            word2id: dictionary mapping words to indices
            unk_id: index for UNK
            id2words: dictionary mapping indices to words
    """
    def __init__(self, word2id=None):
        """ Init Vocab Instance.

        :param word2id: dictionary mapping words to indices
        :type word2id: dict[str, int]
        """
        if word2id:
            self.word2id = word2id
        else:
            self.word2id = dict()
            self.word2id[PAD] = 0  # Pad Token
            self.word2id[UNK] = 1  # Unknown Token
        self.unk_id = self.word2id[UNK]
        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        """ Retrieve word's index. Return the index for unk
        token if the word is out of vocabulary

        :param word: word to look up
        :type word: str
        :returns: index of word
        :rtype: int
        """
        return self.word2id.get(word, self.unk_id)
    
    def __contains__(self, word):
        """ Check if word is captured by Vocab.
        
        :param word: word to look up
        :type word: str
        :returns whether word is in vocab
        :rtype: bool
        """
        return word in self.word2id
    
    def __setitem__(self, key, value):
        """ Raise error, if one tries to edit Vocab directly.
        """
        raise ValueError("Vocab is readonly")
    
    def __len__(self):
        """ Compute number of words in Vocab.
        
        :returns: number of words in Vocab
        :rtype: int
        """
        return len(self.word2id)
    
    def __repr__(self):
        """ Representation of Vocab to be used
        when printing the object.
        """
        return "Vocabulary[size=%d]" % len(self)
    
    def word_from_id(self, wid):
        """ Return mapping of index to word.
        
        :param wid: word index
        :type: int
        :returns: word corresponding to index
        :rtype: str
        """
        return self.id2word[wid]
    
    def add(self, word):
        """ Add word to Vocab, if it is previously unseen.
        
        :param word: to add to Vocab
        :type word: str
        :returns: index that the word has been assigned
        :rtype: int
        """
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]
        
    def save(self, path):
        """ Save Vocab to CSV file, indicated by `path`
        
        :param path: the relative path to the saving file
        :type path: str
        """
        with open(path, "w") as f:
            write = csv.writer(f)
            write.writerows([[w, id] for w, id in self.word2id.items()])

    @staticmethod
    def load(path):
        """ Load Vocab from CSV file, indicated by `path`
        
        :param path: the relative path to the loading file
        :type path: str
        :returns: Vocab instance produced from CSV file
        :rtype: Vocab
        """
        with open(path) as f:
            reader = csv.reader(f)
            return Vocab({row[0]: int(row[1]) for row in reader})

        
    @staticmethod
    def from_corpus(corpus, remove_frac=None, freq_cutoff=None):
        """ Given a corpus, construct a Vocab.
        
        :param corpus: corpus of text produced by read_corpus function
        :type corpus: List[str]
        :param remove_frac: remove len * remove_frac number of words
        :type remove_frac: float
        :param freq_cutoff: if word occurs n < frew_cutoff times, drop the word
        :type freq_cutoff: int
        :returns: Vocab instance produced from provided corpus
        :rtype: Vocab
        """
        vocab_entry = Vocab()
        word_freq = Counter(chain(corpus))
        if freq_cutoff is None:
            freq_cutoff = 1
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
        print("number of word types: {}, number of word types w/ frequency >= {}: {}"
              .format(len(word_freq), freq_cutoff, len(valid_words)))
        top_words = sorted(valid_words, key=lambda word: word_freq[word], reverse=True)
        if remove_frac is not None:
            size = len(top_words) - int(remove_frac * len(top_words))
            top_words = top_words[:size]
            print(f"number of unqiue words retained with remove_frac={remove_frac}: {len(top_words)}")
        for word in top_words:
            vocab_entry.add(word)
        return vocab_entry
    
class LanguageDataset(Dataset):
    """ LanguageDataset is a torch dataset to interact with the Language data.

        Dataset (List[ Tuples[ List[ torch.Tensor ], int ] ]): The vectorized dataset with input and expected output values
        Dataset is an abstract class representing a dataset:
    """
    def __init__(self, context, gov, target, device):
        """ Loads in the context, gov, and target as tensors.

        :param context: context tokens on both sides
        :type context: List[List[int]]
        :param gov: government token
        :type gov: List[int]
        :param target: middle target token
        :type target: List[int]
        """
        self.context = torch.tensor(context).to(device)
        self.gov = torch.tensor(gov).to(device)
        self.target = torch.tensor(target).to(device)
        self.len = len(context)
    
    def __len__(self):
        """ Number of samples in dataset

        :returns: number of samples in dataset
        :rtype: int
        """
        return self.len
    
    def __getitem__(self, index):
        """ The tensor, output for a given index

        :param index: index within dataset
        :type index: int
        :returns: A tuple (x, y, z) where x is the context, y is the govnernment, z is the target
        :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        return self.context[index], self.gov[index], self.target[index]


def data_loaders(preprocessed_data, batch_size=1, shuffle=False, device='cpu'):
    """
    """
    dataset = LanguageDataset(preprocessed_data["context_tokens"], preprocessed_data["gov_tokens"], preprocessed_data["target_tokens"], device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

def create_train_val(data, window_size, samples=50_000):
    """ Create the context, gov, targer data for training and validation

    :param institution_corpus: 
    :type institution_corpus: dict{}
    :param window_size: context window length from both sides of the target word
    :type window_size: int
    :param samples: number of samples collected for each government
    :type samples: int
    """
    
    # Collection holders
    context_tokens = []
    gov_tokens = []
    target_tokens = []

    # Load in tokenized institution corpus
    with open(os.path.join(data, 'institution.csv')) as f:
        reader = csv.reader(f)
        institution_corpus = {row[0]: [int(y) for y in row[1].strip('][').split(', ')] for row in reader}

    pBar = tqdm.tqdm(total=len(institution_corpus) * samples)
    # For each government
    for gov, text in institution_corpus.items():
        count = 0
        length = len(text)

        # Collect samples
        while(count < samples):

            # Sample randomly from the total corpus
            position = math.ceil(random.random() * length)
            if position < (2*window_size + 1):
                position = 2*window_size + 1

            context = text[position - (2*window_size + 1): position - (window_size + 1)] + text[position - window_size: position]
            if len(context) != 2 * window_size:
                continue
            target = text[position - (window_size + 1)]

            # Append data
            context_tokens.append(context)
            gov_tokens.append(gov)
            target_tokens.append(target)

            count += 1
            pBar.update()

    # Make data directory
    os.mkdir(os.path.join(data, f'window{window_size}-samples{samples}'))

    # Train-Validation Split
    split = math.ceil(len(target_tokens) * .92)
    train_range = range(0, split)
    val_range = range(split, len(target_tokens))

    # Shuffle collected data
    random.shuffle(context_tokens)
    random.shuffle(gov_tokens)
    random.shuffle(target_tokens)

    # Write train and validation data
    with open(os.path.join(data, f'window{window_size}-samples{samples}', 'train.csv'), "w") as f:
        write = csv.writer(f)
        write.writerows([[context_tokens[i], gov_tokens[i], target_tokens[i]] for i in train_range])
    
    with open(os.path.join(data, f'window{window_size}-samples{samples}', 'val.csv'), "w") as f:
        write = csv.writer(f)
        write.writerows([[context_tokens[i], gov_tokens[i], target_tokens[i]] for i in val_range])

def get_train_val(data, window_size, samples):
    """ 
    """
    train_data = dict()
    train_data["context_tokens"] = []
    train_data["gov_tokens"] = []
    train_data["target_tokens"] = []

    with open(os.path.join(data, f'window{window_size}-samples{samples}', 'train.csv'), 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            train_data["context_tokens"].append([int(y) for y in row[0].strip('][').split(', ')])
            train_data["gov_tokens"].append(int(row[1]))
            train_data["target_tokens"].append(int(row[2]))

    val_data = dict()
    val_data["context_tokens"] = []
    val_data["gov_tokens"] = []
    val_data["target_tokens"] = []

    with open(os.path.join(data, f'window{window_size}-samples{samples}', 'val.csv'), 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            val_data["context_tokens"].append([int(y) for y in row[0].strip('][').split(', ')])
            val_data["gov_tokens"].append(int(row[1]))
            val_data["target_tokens"].append(int(row[2]))

    return train_data, val_data

def get_freq(target_tokens, vocab_size):
    """ 
    """
    freqs = [1] * vocab_size
    total = len(target_tokens)
    for token in target_tokens:
        freqs[token] += 1
    return list(map(lambda x: total / x, freqs))

###############################################################################
# Query function and classes
###############################################################################

class Query():
    """ 
    """
    def __init__(self, word_vocab, gov_vocab, word_embed, gov_embed, device):
        """ 
        """
        # Set Vocabs
        self.word_vocab = word_vocab
        self.gov_vocab = gov_vocab
        # Set Embeddings
        self.word_embed = word_embed
        self.gov_embed = gov_embed
        # Init query
        self.query_terms = []
        self.query_vec = None

        self.device = device

    def set_query(self, query):
        """ Set query for processing

        :param query: query to process, format -> "(word|gov) [(+|-) (word|gov)]*"
        :type query: str
        """
        split_query = query.split(" ")
        processed_query = []
        curr_sign = 1
        curr_term = split_query[0]
        for term in split_query[1:]:

            match term:

                case '+':
                    
                    if (gov_id := self.gov_vocab[curr_term]) not in [0,1]:  # Not PAD or UNK
                        self.query_terms.append(curr_term)
                        processed_query.append(self.gov_embed(torch.tensor(gov_id, device=self.device)) * curr_sign)
                    elif (word_id := self.word_vocab[curr_term]) not in [0,1]:  # Not PAD or UNK
                        self.query_terms.append(curr_term)
                        processed_query.append(self.word_embed(torch.tensor(word_id, device=self.device)) * curr_sign)
                    else:
                        raise ValueError(f"""Invalid query term has been provide,
                                         {curr_term} is not in word or government vocab.""")
                    curr_sign = 1
                    curr_term = ""

                case '-':

                    if (gov_id := self.gov_vocab[curr_term]) not in [0,1]:  # Not PAD or UNK
                        self.query_terms.append(curr_term)
                        processed_query.append(self.gov_embed(torch.tensor(gov_id, device=self.device)) * curr_sign)
                    elif (word_id := self.word_vocab[curr_term]) not in [0,1]:  # Not PAD or UNK
                        self.query_terms.append(curr_term)
                        processed_query.append(self.word_embed(torch.tensor(word_id, device=self.device)) * curr_sign)
                    else:
                        raise ValueError(f"""Invalid query term has been provide,
                                         {curr_term} is not in word or government vocab.""")
                    curr_sign = -1
                    curr_term = ""

                case _:

                    if curr_term == "":
                        curr_term = term
                    else:
                        curr_term = curr_term + ' ' + term
        
        # Process last term
        if (gov_id := self.gov_vocab[curr_term]) not in [0,1]:  # Not PAD or UNK
            self.query_terms.append(curr_term)
            processed_query.append(self.gov_embed(torch.tensor(gov_id, device=self.device)) * curr_sign)
        elif (word_id := self.word_vocab[curr_term]) not in [0,1]:  # Not PAD or UNK
            self.query_terms.append(curr_term)
            processed_query.append(self.word_embed(torch.tensor(word_id, device=self.device)) * curr_sign)
        else:
            raise ValueError(f"""Invalid query term has been provide,
                                {curr_term} is not in word or government vocab.""")
        
        # lol, sorry that my code is not DRY, "LeT mE aBtRacT iNTo a fUncTiOn", nah bro
        self.query_vec = sum(processed_query)

    def get_words(self, top=5):
        """ 
        """
        cos = torch.nn.CosineSimilarity(dim=0)
        rankings = []
        for word, id in tqdm.tqdm(self.word_vocab.word2id.items()):
            # Exclude words that are in the query
            if word in self.query_terms:
                continue
            score = cos(self.word_embed(torch.tensor(id, device=self.device)), self.query_vec).item()
            rankings.append((word, score))
        
        rankings.sort(reverse=True, key=lambda x: x[1])
        return rankings[:top]

    def get_govs(self, top=5):
        """ 
        """
        cos = torch.nn.CosineSimilarity(dim=0)
        rankings = []
        for gov, id in self.gov_vocab.word2id.items():
            # Exclude words that are in the query
            if gov in self.query_terms:
                continue
            score = cos(self.gov_embed(torch.tensor(id, device=self.device)), self.query_vec).item()
            rankings.append((gov, score))
        
        rankings.sort(reverse=True, key=lambda x: x[1])
        return rankings[:top]