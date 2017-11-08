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

def load_character_data(fname, char_to_id, max_sent_len, max_word_len=32):
    X = []

    with open(fname, 'r', encoding='utf-8') as in_f:
        for line in in_f:
            sent_id, sentence, category, intensity = line.strip().split('\t')
            if args.bytes:
                char_ids = [char_to_id[byte] for char in sentence for byte in map(ord, char.encode('utf8'))]
            else:
                char_ids = [char_to_id[char] for char in sentence]

            X.append([char_to_id[SENT_START]]+char_ids+[char_to_id[SENT_END]])

    return X
    # sent_lens = [len(s) for s in X]
    # max_sent_len_data = max(sent_lens)
    # percentile = int(np.percentile(sent_lens, 90))
    #
    # # cutoff for max_sent_len
    # discarded_tokens = 0
    # used_tokens = 0
    # cut_X = []
    # for sent in X:
    #     discarded_tokens += max(len(sent) - max_word_len, 0)
    #     used_tokens += len(sent[:max_word_len])
    #     cut_X.append(sent[:max_word_len])
    #
    # if __debug__:
    #     print('max len in dataset: {0}\t90-percentile: {2}\tmax len used: {1}'.format(max_sent_len_data, max_word_len, percentile))
    #     print('n discarded toks:\t{0}'.format(discarded_tokens))
    #     print('n used toks:\t\t{0}'.format(used_tokens))
    #     print('prop used toks:\t\t{0}'.format(used_tokens / float(used_tokens+discarded_tokens)))
    #
    # return cut_X

def load_word_data(fname, word_to_id, tag_to_id, max_sent_len, is_training=False):
    """
    Reading of CoNLL file, converting to word ids.
    If is_training, unknown mwes will be added to the embedding dictionary using a heuristic.
    """
    print("reading ",fname)
    X, y = [], []

    with open(fname, 'r', encoding='utf-8') as in_f:
        curr_X, curr_y = [], []
        for line in in_f:
            #line = line.strip()
            #if len(line) == 0: continue

            sent_id, sentence, category, intensity = line.strip().split('\t')
            for token in sentence.split(): # TODO: Real tokenisation
                if not args.words:
                    curr_X.append(word_to_id[UNKNOWN])
                # Some preprocessing
                if re.match('^[0-9\.\,-]+$', token):
                    curr_X.append(word_to_id[NUMBER])
                else:
                    #token = ''.join(ch for ch in token if ch not in exclude)
                    if is_training and token.lower() in word_to_id:# and args.mwe and ('~' in token or '-' in token):
                        curr_X.append(word_to_id[token.lower()])
                        print(token)
                        #curr_X.append(attempt_reconstruction(token, word_to_id))
                    else:
                        #print("unk*****", token) #if token not in embeddings it's UNK (or mwu if option off)
                        curr_X.append(word_to_id[UNKNOWN])

            #curr_y.append(tag_to_id[tag])

            #if args.shorten_sents and len(curr_X) >= max_sent_len-2:
            X.append(curr_X)
            y.append(float(intensity))
            curr_X = []#word_to_id[SENT_CONT]]
                #curr_y = []#tag_to_id[SENT_CONT]]
    ## get some stats on dataset
    sent_lens = [len(s) for s in X]
    max_sent_len_data = max(sent_lens)
    percentile = int(np.percentile(sent_lens, 90))
    ## max sentence cutoff
    # Two options: either discard sentences (as in earlier code), or use all up to max_sent_len (
    # Leave room for padding
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
        filtered_word_to_id = defaultdict(lambda: len(filtered_word_to_id))
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
    #y = [s[:max_sent_len-2] for s in y]

    if __debug__:
        print('max len in dataset: {0}\t90-percentile: {2}\tmax len used: {1}'.format(max_sent_len_data, max_sent_len, percentile))
        print('n discarded sents: {0}'.format(old_len - len(X)))
        print('n discarded toks: {0}'.format(discarded_tokens))

    if args.bookkeeping:
        dsetname = os.path.basename(fname).rstrip('.conllu')
        save_ids(word_to_id, tag_to_id, dsetname)

    return X, y, word_to_id, tag_to_id

def save_ids(word_to_id, tag_to_id, fname):
    write_mapping(word_to_id, experiment_dir+'/{0}_word2id.txt'.format(fname))
    write_mapping(tag_to_id, experiment_dir+'/{0}_tag2id.txt'.format(fname))
