###############################################################################
#  Library imports
###############################################################################

import argparse
import os
import sys
import csv

import util

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
# Parse command inputs
###############################################################################

parser = argparse.ArgumentParser(description='PyTorch Policy Gov2Vec Embeddings Model - Data Initializer')

parser.add_argument('--data', type=str, default='./data/govtext',
                    help='location to save initialized data')
parser.add_argument('--window', type=int, default=10,
                    help='context window length from both sizes of the target word')
parser.add_argument('--samples', type=int, default=50_000,
                    help='number of samples collected for training')
parser.add_argument('--freq-cutoff', type=int, default=10, metavar='FC',
                    help='frequency cutoff for words to be added to the vocab')

args = parser.parse_args()

###############################################################################
# Initialize data
###############################################################################

# Open institution corpora 
institution_corpora = dict()

with open('./data/clean-data/institution_corpora.csv', 'r') as f:
    reader = csv.reader(f)
    institution_corpora.update({row[0]: row[1] for row in reader})

# Consolidate institutions and corpora
corpora = []
govs = []

for gov, corpus in institution_corpora.items():
    corpora.append(corpus)
    govs.append(gov)

total_corpus = ' '.join(corpora).split(' ')  # List of all words

# Create vocab
word_vocab = util.Vocab.from_corpus(corpus=total_corpus, freq_cutoff=args.freq_cutoff)
gov_vocab = util.Vocab.from_corpus(corpus=govs)

# Save vocab
word_vocab.save(os.path.join(args.data, "words.csv"))
gov_vocab.save(os.path.join(args.data, "govs.csv"))

# Tokenize data
tokenized_data = [
    [gov_vocab[gov], [word_vocab[w] for w in corpus.split(" ") if word_vocab[w] not in [0,1]]] 
    for gov, corpus in institution_corpora.items()
]

# Save tokenized data
with open(os.path.join(args.data, 'institution.csv'), 'w') as f:
    writer = csv.writer(f)
    writer.writerows(tokenized_data)

# Check if the specified data has been created
if f'window{args.window}-samples{args.samples}' not in os.listdir(args.data):
    util.create_train_val(args.data, args.window, args.samples)