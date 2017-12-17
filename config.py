#!/usr/bin/env python

import os
import time
import codecs
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--train', help='Train set', nargs='+')
parser.add_argument('--dev', help='Dev set', nargs='+')
parser.add_argument('--test', help='Test set', nargs='+')
parser.add_argument('--aux', help='Aux class set', nargs='+')
parser.add_argument('--embeddings', help='Pretrained embeddings', default='./embeddings.txt', type=str)
parser.add_argument('--ignore-embeddings', help='Ignore pretrained embeddings', action='store_true')
parser.add_argument('--multilingual', help='Multilingual embeddings', action='store_true')
parser.add_argument('--freeze', help='Freeze embedding weights', default=False, type=bool)
parser.add_argument('--word-embedding-dim', help='Only if not using pretrained', type=int, default=64)
parser.add_argument('--char-embedding-dim', type=int, default=128)
parser.add_argument('--rnn', help='Use RNN after convolutions', action='store_true')
parser.add_argument('--rnn-dim', help='RNN dim', type=int, default=128)
parser.add_argument('--epochs', help='n epochs', type=int, default=5)
parser.add_argument('--bsize', help='batch size', type=int, default=10)
parser.add_argument('--save', help='save model', action='store_true')
parser.add_argument('--chars', help='use characters', action='store_true')
parser.add_argument('--bytes', help='use bytes', action='store_true')
parser.add_argument('--words', help='use words', action='store_true')
parser.add_argument('--nwords', help='use max n words', type=int, default=100000)
parser.add_argument('--tag', help='extra experiment nametag', type=str)
parser.add_argument('--resnet', help='use resnets', type=int, default=0)
parser.add_argument('--max-word-len', help='max length word for char embeds', default=256, type=int)
parser.add_argument('--max-sent-len', help='max length word sequence', default=60, type=int)
parser.add_argument('--early-stopping', help='use early stopping callback', type=int)
parser.add_argument('--bn', help='use batch normalisation', action='store_true')
parser.add_argument('--dropout', help='use dropout', type=float, default=0.0)
parser.add_argument('--verbose', help='keras verbosity', type=int, default=1)
parser.add_argument('--reuse', help='model path', type=str)
parser.add_argument('--plot', help='plot model', action='store_true')
parser.add_argument('--kfold', help='amount of folds in validation', type=int)


args = parser.parse_args()

# Values for embedding paddings etc.
SENT_START = '<w>'
SENT_END = '</w>'
SENT_PAD = '<PAD>'
SENT_CONT = '##'
UNKNOWN = '_UNK'
NUMBER = '####'

if args.embeddings:
    emb_name = os.path.basename(args.embeddings)
else:
    emb_name = ''

if args.bytes:
    args.chars = True

# Logging name / path
experiment_tag = 'ep-{0}_bsize-{1}_emb-{2}_train-{3}_dev-{4}_test-{5}_words-{6}_chars-{7}_resnet-{8}_time-{9}'.format(
args.epochs,
args.bsize,
emb_name,
os.path.basename(args.train[0]),
os.path.basename(args.dev[0]),
os.path.basename(args.test[0]),
args.words,
args.chars,
args.resnet,
time.time()
)
if args.tag:
    experiment_tag += '_tag-{0}'.format(args.tag)

if not args.chars and not args.words:
    print("need to specify (either or both): --chars --words")
    exit()
