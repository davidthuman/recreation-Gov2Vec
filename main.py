# coding: utf-8

###############################################################################
#  Library imports
###############################################################################

import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
import torch.optim as optim

import model as _model
import util

###############################################################################
# Parse command inputs
###############################################################################

parser = argparse.ArgumentParser(description='PyTorch Policy Gov2Vec Embeddings Model - Training and Testing')

parser.add_argument('--data', type=str, default='./data/govtext',
                    help='location of the data corpus')
parser.add_argument('--window', type=int, default=10,
                    help='context window length from both sizes of the target word')
parser.add_argument('--samples', type=int, default=50_000,
                    help='number of samples collected for training')
parser.add_argument('--arch-context', type=str, default='AVG',  metavar='CONTEXT',
                    help='architecture of context combination (AVG, SUM, CONCAT)')
parser.add_argument('--arch-gov', type=str, default='AVG',  metavar='GOV',
                    help='architecture of government combination (AVG, SUM, CONCAT)')
parser.add_argument('--emsize', type=int, default=100,
                    help='size of word and gov embeddings')
parser.add_argument('--lr', type=float, default=0.1,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=25,
                    help='upper epoch limit')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='batch size')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='use CUDA')
parser.add_argument('--mps', action='store_true', default=False,
                        help='enables macOS GPU training')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
parser.add_argument('--test-run', action='store_true',
                    help='run testing on the current model')
parser.add_argument('--dry-run', action='store_true',
                    help='verify the code and the model')

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
# Load data
###############################################################################

# Check if the specified data has been created
if f'window{args.window}-samples{args.samples}' not in os.listdir(args.data):
    util.create_train_val(args.data, args.window, args.samples)

# Create training and validation sets
train_data, val_data = util.get_train_val(args.data, args.window, args.samples)

# Load in Vocab
word_vocab = util.Vocab.load(os.path.join(args.data, 'words.csv'))
gov_vocab = util.Vocab.load(os.path.join(args.data, 'govs.csv'))

# Create training and validation loaders
train_loader = util.data_loaders(train_data, args.batch_size, True, device)
val_loader = util.data_loaders(val_data, args.batch_size, True, device)

###############################################################################
# Build the model
###############################################################################

loss_weights = torch.tensor(
    util.get_freq(train_data['target_tokens'], len(word_vocab)),
    device=device
)

model = _model.Gov2Vec_Model(
    arch_context=args.arch_context,
    arch_gov=args.arch_gov,
    vocab_size=len(word_vocab), 
    gov_size=len(gov_vocab),
    embed_dim=args.emsize,
    window_size=args.window,
    loss_weights=loss_weights
).to(device)

###############################################################################
# Training code
###############################################################################

def evaluate(val_loader):
    """ Evaluate the validation data

    :param val_loader: data loader for the validation data
    :type val_loader: DataLoader
    :returns: loss for the validation dataset
    :rtype: float
    """
    model.eval()  # turn on evaluation mode which disables dropout.
    total_loss = 0.
    for (context_batch, gov_batch, target_batch) in val_loader:
        outputs = model(context_batch, gov_batch)
        total_loss += model.criterion(outputs, target_batch).item()
    return total_loss / len(val_loader)

def train(train_loader):
    """ Trains a single epoch of the training dataset

    :param train_loader: data loader for the training data
    :type train_loader: DataLoader
    """
    model.train()  # Turn on training mode which enables dropout.
    prev_loss = 1_000_000
    total_loss = 0.
    start_time = time.time()
    for batch, (context_batch, gov_batch, target_batch) in enumerate(train_loader):
        model.zero_grad()
        output = model(context_batch, gov_batch)
        loss = model.criterion(output, target_batch)
        loss.backward()

        # Optimizer AdamW implementation to actively change learning rate should be considered
        for p in model.parameters():
            p.data.add_(p.grad, alpha=-lr)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            curr_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_loader), lr,
                elapsed * 1000 / args.log_interval, curr_loss, math.exp(curr_loss)))
            prev_loss = total_loss
            total_loss = 0
            start_time = time.time()
        if args.dry_run:
            break

def export_onnx(path, batch_size, window_len):
    """ Create and save an ONNX files for the model

    :param path: path to save the files
    :type path: str
    :param batch_size:
    :type batch_size: int
    :param window_len: 
    :type window_len: int
    """
    print('The model is also exported in ONNX format at {}.'.format(os.path.realpath(args.onnx_export)))
    model.eval()
    for (context, gov, tagret) in val_loader:
        break
    context = context[[0],:]
    gov = gov[[0]]
    torch.onnx.export(model, (context, gov), path)

###############################################################################
# Loop over epochs
###############################################################################

lr = args.lr
best_val_loss = None

if not args.test_run:
    # At any point you can hit Ctrl + C to break out of the training early.
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train(train_loader=train_loader)
            val_loss = evaluate(val_loader=val_loader)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                            val_loss, math.exp(val_loss)))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            LOSS_TOL = 0.05  # tolerance for validation loss improvement
            if not best_val_loss or (best_val_loss - val_loss > LOSS_TOL):
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 4.0
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)

if len(args.onnx_export) > 0:
    # Export the model in ONNX format.
    export_onnx(args.onnx_export, batch_size=2, window_len=args.window)

###############################################################################
# Test queries
###############################################################################

test_queries = {
    "Branch 1": "candidate + elected + campaign",
    "Branch 2": "long + term + government + career",
    "Branch 3": "rule + precedent + interpret",
    "Branch 4": "validity + truth",
    "Branch 5": "statistics + science + data - story - anecdote",
    "Branch 6": "order + direct - contemplate - consider",
    "113th House Economic": "climate + emissions + House 113th - Barack Obama + economy - environment",
    "113th House Environmental": "climate + emissions + House 113th - Barack Obama - economy + environment",
    "Obama Economic": "climate + emissions - House 113th + Barack Obama + economy - environment",
    "Obama Environmental": "climate + emissions - House 113th + Barack Obama - economy + environment",
    "106th House Oil": "war + House 106th - House 107th + oil - terrorism",
    "106th House Terrorism": "war + House 106th - House 107th - oil + terrorism",
    "107th House Oil": "war - House 106th + House 107th + oil - terrorism",
    "107th House Terrorism": "war - House 106th + House 107th - oil + terrorism",
}

query_processer = util.Query(word_vocab, gov_vocab, model.word_embedding, model.gov_embedding, device)

for title, query in test_queries.items():

    try:
        query_processer.set_query(query)
    except ValueError:
        print(str(ValueError))
        continue

    words = query_processer.get_words()
    govs = query_processer.get_govs()

    print(f"{title}: {query}")
    print()
    print(f"Top 5 Words")
    print('-' * 89)
    for word, score in words:
        print(f"\t{word}: {score}")
    print(f"Top 5 Institutions")
    print('-' * 89)
    for gov, score in govs:
        print(f"\t{gov}: {score}")

