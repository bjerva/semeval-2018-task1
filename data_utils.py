#!/usr/bin/env python

#import h5py
import utils
from sys import argv
import random
import numpy as np
from keras.preprocessing import sequence
from collections import defaultdict
from config import *
import pickle
np.random.seed(1337)
random.seed(1337)

def preprocess_words(X_ids, Y_ids, word_to_id, tag_to_id, word_vectors, max_sent_len, is_training=False):
    if __debug__: print('Reshaping data...')

    id_to_word = dict((val, key) for key, val in word_to_id.items())
    nb_classes = len(tag_to_id)

    X = []
    for idx, sentence in enumerate(X_ids):
        curr_X = np.zeros((len(sentence)+2, ), dtype=np.int32) # +2 for sent_start, sent_end
        curr_X[0] = word_to_id[SENT_START]
        for idy, word_id in enumerate(sentence):
            curr_X[idy+1] = word_id # +1 for offset
        curr_X[-1] = word_to_id[SENT_END]
        X.append(curr_X)
    y = np.zeros((len(Y_ids), nb_classes), dtype=np.int32)#[]
    y = Y_ids
    X = sequence.pad_sequences(X, maxlen=max_sent_len, dtype=np.int32, value=word_to_id[SENT_PAD])

    return X, y, word_vectors

def read_word_data(trainfiles, devfiles, testfiles, aux, word_to_id, word_vectors, max_sent_len):
    """
    Load data from CoNLL file and convert to ids (utils load function)
    Preprocess data
    TODO: hdf5 data caching
    """

    X_train = []
    y_train = []

    X_dev = []
    y_dev = []

    X_test = []
    y_test = []

    train_lengths = []
    dev_lengths = []
    test_lengths = []

    prev = 0

    tag_to_id = defaultdict(lambda: len(tag_to_id))
    if args.ignore_embeddings and word_to_id == {}: #skal begge flag være sat af?
        word_to_id = defaultdict(lambda: len(word_to_id))
    filtered_word_to_id = defaultdict(lambda: len(filtered_word_to_id))

    for auxfiles in aux:
        (_, _, _, _, _, y_dict) = utils.load_word_data(auxfiles, word_to_id, filtered_word_to_id, tag_to_id, max_sent_len)

    for trainfile in trainfiles:
        (X_train_ids, y_train_ids, new_filtered_word_to_id, tag_to_id, length, y_dict) = utils.load_word_data(trainfile, word_to_id, filtered_word_to_id, tag_to_id, max_sent_len, y_dict=y_dict, is_training=True)
        X_train_temp, y_train_temp, word_vectors = preprocess_words(X_train_ids, y_train_ids, new_filtered_word_to_id, tag_to_id, word_vectors, max_sent_len)
        
        filtered_word_to_id.update(new_filtered_word_to_id)

        train_lengths.append(length + prev)
        prev += length
        X_train.extend(X_train_temp)
        y_train.extend(y_train_temp)
    
    prev = 0
    #word_to_id = filtered_word_to_id

    for devfile in devfiles:
        (X_dev_ids, y_dev_ids,_,_, length, _) = utils.load_word_data(devfile, word_to_id, filtered_word_to_id, tag_to_id, max_sent_len, y_dict=y_dict)
        X_dev_temp, y_dev_temp, _ = preprocess_words(X_dev_ids, y_dev_ids, word_to_id, tag_to_id, word_vectors, max_sent_len)

        dev_lengths.append(length + prev)
        prev += length
        X_dev.extend(X_dev_temp)
        y_dev.extend(y_dev_temp)

    prev = 0
    if testfiles:
        for testfile in testfiles:
            (X_test_ids, y_test_ids,_,_, length, _) = utils.load_word_data(testfile, word_to_id, filtered_word_to_id, tag_to_id, max_sent_len)
            X_test_temp, y_test_temp, word_vectors = preprocess_words(X_test_ids, y_test_ids, word_to_id, tag_to_id, word_vectors, max_sent_len)

            test_lengths.append(length + prev)
            prev += length
            X_test.extend(X_test_temp)
            y_test.extend(y_test_temp)
    else:
        X_test, y_test  = None, None

    return (X_train, y_train), (X_dev, y_dev), (X_test, y_test), word_vectors, word_to_id, train_lengths, dev_lengths, test_lengths

def read_char_data(trainfiles, devfiles, testfiles, char_to_id, max_sent_len, max_word_len):
    """
    read char data, preprocess
    """

    X_train = []
    X_dev = []
    X_test = []

    for trainfile in trainfiles:
        print('Reading train chars')
        X_train_ids = utils.load_character_data(trainfile, char_to_id, max_sent_len, max_word_len)
        X_train_temp = preprocess_chars(X_train_ids, char_to_id, max_sent_len, max_word_len)

        X_train.extend(X_train_temp)

    for devfile in devfiles:
        print('Reading dev chars')
        X_dev_ids = utils.load_character_data(devfile, char_to_id, max_sent_len, max_word_len)
        X_dev_temp = preprocess_chars(X_dev_ids, char_to_id, max_sent_len, max_word_len)

        X_dev.extend(X_dev_temp)

    if testfiles:
        for testfile in testfiles:
            print('Reading test chars')
            X_test_ids = utils.load_character_data(testfile, char_to_id, max_sent_len, max_word_len)
            X_test_temp = preprocess_chars(X_test_ids, char_to_id, max_sent_len, max_word_len)

            X_test.extend(X_test_temp)
    else:
        X_test = None

    return X_train, X_dev, X_test

def preprocess_chars(X_ids, char_to_id, max_sent_len, max_word_len):
    X = []
    for idx, sentence in enumerate(X_ids):
        curr_X = []

        curr_sent = np.zeros((len(sentence)+2, ), dtype=np.int16) # +2 for word_start, word_end
        curr_sent[0] = char_to_id[SENT_START]
        for idz, char_id in enumerate(sentence):
            curr_sent[idz+1] = char_id
        curr_sent[-1] = char_to_id[SENT_END]

        curr_X = list(curr_sent)

        while len(curr_X) < max_sent_len: # Extra fill-padding before actual words
            curr_X = [char_to_id[SENT_PAD]] + list(curr_X)

        X.append(curr_X)

    X = sequence.pad_sequences(X, maxlen=max_word_len, dtype=np.int16, value=0)

    return X
