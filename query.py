# Implement querying on model

import argparse
import torch
import os

import data
import util

###############################################################################
# Parse command inputs
###############################################################################

parser = argparse.ArgumentParser(description='PyTorch Policy Gov2Vec Embeddings Model')

parser.add_argument('--data', type=str, default='./data/govtext',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='./terms.txt',
                    help='output file for terms')
parser.add_argument('--query', type=str,
                    help='query to ask model, format": "(word|gov) [(+|-) (word|gov)]*"')
parser.add_argument('--top', type=int, default=5,
                    help='number of similar to gather')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='use CUDA')
parser.add_argument('--mps', action='store_true', default=False,
                        help='enables macOS GPU training')

args = parser.parse_args()

###############################################################################
# Set the random seed manually for reproducibility
###############################################################################

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda.")
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    if not args.mps:
        print("WARNING: You have mps device, to enable macOS GPU run with --mps.")

use_mps = args.mps and torch.backends.mps.is_available()
if args.cuda:
    device = torch.device("cuda")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

###############################################################################
# Load model
###############################################################################

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f, map_location=device)
model.eval()

###############################################################################
# Load data
###############################################################################

word_vocab = data.Vocab.load(os.path.join(args.data, 'words.csv'))
gov_vocab = data.Vocab.load(os.path.join(args.data, 'govs.csv'))

###############################################################################
# Process query
###############################################################################

query = util.Query(word_vocab, gov_vocab, model.word_embedding, model.gov_embedding, device)
query.set_query(args.query)

words = query.get_words(args.top)
govs = query.get_govs(args.top)

with open(args.outf, 'w') as f:
    
    f.write(f"Query: {args.query}\n")
    f.write('\n')
    f.write(f"Top {args.top} Words\n")
    f.write(('-' * 89) + '\n')
    for word, score in words:
        f.write(f"\t{word}: {score}\n")
    f.write('\n')
    f.write(f"Top {args.top} Institutions\n")
    f.write(('-' * 89) + '\n')
    for gov, score in govs:
        f.write(f"\t{gov}: {score}\n")