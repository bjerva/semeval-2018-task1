#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import re
import string
import codecs
import numpy as np
from collections import defaultdict, Counter
from config import *

from codecs import open

bad_chars = [',', '.', '!', ' ']

def read_word_embeddings(fname):
    word_vec_map = {}
    word_id_map = {}
    with open(fname, 'r', encoding='utf-8') as in_f:
        for idx, line in enumerate(in_f):
            fields = line.strip().split()
            word = fields[0]
            embedding = np.asarray([float(i) for i in fields[1:]], dtype=np.float32)

            word_vec_map[word] = embedding
            word_id_map[word] = len(word_id_map)

    # get dimensions from last word added
    vec_dim = len(word_vec_map[word])
    return word_vec_map, word_id_map, vec_dim

def load_character_data(fname, char_to_id, max_sent_len, max_word_len=32, is_aux=False):
    X = []

    with open(fname, 'r', encoding='utf-8') as in_f:
        next(in_f)
        for line in in_f:
            content = line.strip().split('\t')
            sent_id, sentence = content[0:2]
            labels = content[2:]
            if args.bytes:
                char_ids = [char_to_id[byte] for char in sentence for byte in map(ord, char.encode('utf8'))]
            else:
                char_ids = [char_to_id[char] for char in sentence]
            X.append(char_ids)
    return X

def load_word_data(fname, word_to_id, filtered_word_to_id, tag_to_id, max_sent_len, y_dict={}, is_training=False):
    """
    Reading of CoNLL file, converting to word ids.
    If is_training, unknown mwes will be added to the embedding dictionary using a heuristic.
    """
    print("reading ",fname)
    X, y = [], []
    length = 0
    reg_no_class = 0
    with open(fname, 'r', encoding='utf-8') as in_f:
        next(in_f)
        curr_X, curr_y = [], []
        for line in in_f:
            length +=1
            content = line.strip().split('\t')
            sent_id, sentence = content[0:2]
            labels = content[2:]
            for token in sentence.split():
                for char in bad_chars:
                    token = token.replace(char, '')
                if not args.words:
                    curr_X.append(word_to_id[UNKNOWN])
                if re.match('^[0-9\.\,-]+$', token):
                    curr_X.append(word_to_id[NUMBER])
                elif token.startswith('@'):
                    curr_X.append(word_to_id[USER])
                else:
                    if token in word_to_id:
                        curr_X.append(word_to_id[token])
                    elif args.ignore_embeddings:
                        curr_X.append(word_to_id[token.lower()])
                    #elif (token.lower() not in word_to_id) and is_training:
                        #curr_X.append(word_to_id[token.lower()])
                    else:
                        curr_X.append(word_to_id[UNKNOWN])
            X.append(curr_X)
            if len(labels) <= 2:
                floats = [float(labels[1])]
                try:
                    floats.extend(y_dict[sent_id])
                except KeyError:
                    floats.extend([-1.0]*11)
                    reg_no_class += 1
                y.append(floats)
            else:
                floats = [float(x) for x in labels]
                y_dict[sent_id] = floats 
            curr_X = []

    sent_lens = [len(s) for s in X]
    max_sent_len_data = max(sent_lens)
    percentile = int(np.percentile(sent_lens, 90))

    old_len = len(X)
    discarded_tokens = 0
    total_toks = 0
    for idx, s in enumerate(X):
        total_toks += len(s)
        if len(s) > max_sent_len-2:
            discarded_tokens += len(s) - (max_sent_len - 2)

    if is_training:
        to_use = set([key for key, val in Counter([word for sentence in X for word in sentence]).most_common(args.nwords)])
        id_to_word = dict([(value, key) for key, value in word_to_id.items()])
        word_filtered = []
        for sentence in X:
            filtered_sent = []
            for word_id in sentence:
                if word_id in to_use:
                    filtered_sent.append(filtered_word_to_id[id_to_word[word_id]])
            word_filtered.append(filtered_sent)
        word_to_id = filtered_word_to_id
        X = word_filtered

    X = [s[:max_sent_len-2] for s in X]

    if __debug__:
        print('max len in dataset: {0}\t90-percentile: {2}\tmax len used: {1}'.format(max_sent_len_data, max_sent_len, percentile))
        print('n discarded sents: {0}'.format(old_len - len(X)))
        print('n discarded toks: {0}'.format(discarded_tokens))
        print('amount of tweets with no class labels: {0}\n'.format(reg_no_class))

    return X, y, word_to_id, filtered_word_to_id, tag_to_id, length, y_dict