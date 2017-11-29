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

def write_features_hdf5(train, valid, test):
    f = h5py.File(args.hdf5, "w")

    train_grp = f.create_group("train")
    train_x = train_grp.create_dataset("train_x", train[0].shape, dtype='f', compression="gzip", compression_opts=9)
    train_y = train_grp.create_dataset("train_y", train[1].shape, dtype='i', compression="gzip", compression_opts=9)

    valid_grp = f.create_group("valid")
    valid_x = valid_grp.create_dataset("valid_x", valid[0].shape, dtype='f', compression="gzip", compression_opts=9)
    valid_y = valid_grp.create_dataset("valid_y", valid[1].shape, dtype='i', compression="gzip", compression_opts=9)

    test_grp = f.create_group("test")
    test_x = test_grp.create_dataset("test_x", test[0].shape, dtype='f', compression="gzip", compression_opts=9)
    test_y = test_grp.create_dataset("test_y", test[1].shape, dtype='i', compression="gzip", compression_opts=9)

    train_x.write_direct(np.ascontiguousarray(train[0], dtype=train[0].dtype))
    train_y.write_direct(np.ascontiguousarray(train[1], dtype=train[1].dtype))
    valid_x.write_direct(np.ascontiguousarray(valid[0], dtype=valid[0].dtype))
    valid_y.write_direct(np.ascontiguousarray(valid[1], dtype=valid[1].dtype))
    test_x.write_direct(np.ascontiguousarray(test[0], dtype=test[0].dtype))
    test_y.write_direct(np.ascontiguousarray(test[1], dtype=test[1].dtype))

    f.close()

def load_hdf5():
    f = h5py.File(args.hdf5, 'r')
    sets = ('train', 'valid', 'test')
    parts = ('x', 'y')

    hdf5s = [f['{0}/{0}_{1}'.format(i, j)] for i in sets for j in parts]

    np_arrays = [np.zeros(h5.shape, h5.dtype) for h5 in hdf5s]
    for idx, h5 in enumerate(hdf5s):
        h5.read_direct(np_arrays[idx])
    f.close()

    return ((np_arrays[0], np_arrays[1]),
            (np_arrays[2], np_arrays[3]),
            (np_arrays[4], np_arrays[5]))


def preprocess_words(X_ids, Y_ids, word_to_id, tag_to_id, word_vectors, max_sent_len, is_training=False):
    if __debug__: print('Reshaping data...')

    id_to_word = dict((val, key) for key, val in word_to_id.items())
    nb_classes = len(tag_to_id)

    X = []
    for idx, sentence in enumerate(X_ids):
        curr_X = np.zeros((len(sentence)+2, ), dtype=np.int32) # +2 for sent_start, sent_end
        curr_X[0] = word_to_id[SENT_START]
        #XXX: Some bug here with words
        for idy, word_id in enumerate(sentence):
            if is_training and args.mwe and type(word_id) == list:
                vectors = np.asarray([word_vectors[id_to_word[i]] for i in word_id[:-1]])
                compositional_rep = np.sum(vectors, axis=0, dtype=np.float32)
                word_id = word_id[-1]
                word_vectors[id_to_word[word_id]] = compositional_rep

            curr_X[idy+1] = word_id # +1 for offset
        curr_X[-1] = word_to_id[SENT_END]
        X.append(curr_X)
    y = np.zeros((len(Y_ids), nb_classes), dtype=np.int32)#[]
    '''
    for idx, tag_id in enumerate(Y_ids):
        #curr_Y = np.zeros((len(sentence)+2, nb_classes), dtype=np.int32) # +2 for sent_start, sent_end
        #curr_Y[0, tag_to_id[SENT_START]] = 1
        #for idy, tag_id in enumerate(sentence):
        y[idx, tag_id] = 1 # +1 for offset
        #curr_Y[-1, tag_to_id[SENT_END]] = 1
        #y.append(curr_Y)
    '''
    y = Y_ids
    X = sequence.pad_sequences(X, maxlen=max_sent_len, dtype=np.int32, value=word_to_id[SENT_PAD])

    return X, y, word_vectors

def read_word_data(trainfile, devfile, testfile, auxfile, word_to_id, word_vectors, max_sent_len):
    """
    Load data from CoNLL file and convert to ids (utils load function)
    Preprocess data
    TODO: hdf5 data caching
    """
    tag_to_id = defaultdict(lambda: len(tag_to_id))
    if args.ignore_embeddings and word_to_id == {}: #skal begge flag være sat af?
        word_to_id = defaultdict(lambda: len(word_to_id))

    (X_train_ids, y_train_ids, word_to_id, tag_to_id) = utils.load_word_data(trainfile, word_to_id, tag_to_id, max_sent_len, is_training=True)
    (X_dev_ids, y_dev_ids, _,_) = utils.load_word_data(devfile, word_to_id, tag_to_id, max_sent_len)

    if testfile:
        (X_test_ids, y_test_ids, _,_) = utils.load_word_data(testfile, word_to_id, tag_to_id, max_sent_len)
        X_test, y_test, word_vectors = preprocess_words(X_test_ids, y_test_ids, word_to_id, tag_to_id, word_vectors, max_sent_len)
    else:
        X_test, y_test  = None, None
    
    if auxfile:
        (X_aux_ids, y_aux_ids, _,_) = utils.load_word_data(auxfile[0], word_to_id, tag_to_id, max_sent_len, is_aux=True, is_training=True)
        X_aux, y_aux, _ = preprocess_words(X_aux_ids, y_aux_ids, word_to_id, tag_to_id, word_vectors, max_sent_len)
    else:
        X_aux, y_aux  = None, None

    # padding
    X_train, y_train, word_vectors = preprocess_words(X_train_ids, y_train_ids, word_to_id, tag_to_id, word_vectors, max_sent_len)
    X_dev, y_dev, _ = preprocess_words(X_dev_ids, y_dev_ids, word_to_id, tag_to_id, word_vectors, max_sent_len)

    return (X_train, y_train), (X_dev, y_dev), (X_test, y_test), (X_aux, y_aux), word_vectors, word_to_id

def read_char_data(trainfile, devfile, testfile, auxfile, char_to_id, max_sent_len, max_word_len):
    """
    read char data, preprocess
    """
    X_train_ids = utils.load_character_data(trainfile, char_to_id, max_sent_len, max_word_len)
    X_dev_ids = utils.load_character_data(devfile, char_to_id, max_sent_len, max_word_len)

    if testfile:
        X_test_ids = utils.load_character_data(testfile, char_to_id, max_sent_len, max_word_len)
        X_test = preprocess_chars(X_test_ids, char_to_id, max_sent_len, max_word_len)
    else:
        X_test = None

    if testfile:
        X_aux_ids = utils.load_character_data(auxfile, char_to_id, max_sent_len, max_word_len, is_aux=True)
        X_aux = preprocess_chars(X_aux_ids, char_to_id, max_sent_len, max_word_len)
    else:
        X_aux = None

    print("preprocess chars")
    X_train = preprocess_chars(X_train_ids, char_to_id, max_sent_len, max_word_len)
    X_dev = preprocess_chars(X_dev_ids, char_to_id, max_sent_len, max_word_len)

    return X_train, X_dev, X_test, X_aux

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
