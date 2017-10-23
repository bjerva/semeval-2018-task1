#!/usr/bin/env python

import numpy as np
import pickle
from sys import argv
from config import *

def write_confusion_matrix(predicted, gold, nb_classes):
    predicted = np.ravel(np.asarray([np.argmax(i, axis=1) for i in predicted]))
    gold = np.ravel(gold)
    conf_mat = np.zeros((nb_classes, nb_classes), dtype=np.int32)

    for pred, exp in zip(predicted, gold):
        conf_mat[pred][exp] += 1

    np.save(experiment_dir+'/conf_mat.np', conf_mat)

def prepare_error_analysis(X_ids, predicted, gold, vocab_size, fname):
    error_mat = np.zeros((vocab_size, 2), dtype=np.uint64)
    for idx in range(X_ids.shape[0]):
        for idy in range(X_ids.shape[1]):
            correct = 1 if np.argmax(predicted[idx, idy]) == gold[idx, idy] else 0
            error_mat[X_ids[idx, idy]][correct] += 1

    np.save(experiment_dir+'/error_analysis.np', error_mat)

def read_index_dict(fname):
    with open(fname, 'rb') as in_f:
        return pickle.load(in_f)

def make_error_analysis(error_mat, word_indices):
    props = error_mat[:, 0] / np.sum(-error_mat, axis=1)
    cutoff_idx = [idx for idx, i in enumerate(props) if abs(i) > min_freq]

    relevant = error_mat[cutoff_idx]
    sorted_idx = relevant.argsort()

    for idx in sorted_idx:
        corr_err = relevant[idx]
        word = word_indices[cutoff_idx[relevant[idx]]]

    import pdb; pdb.set_trace()

if __name__ == '__main__':
    min_freq = 0.5 if len(argv) <= 3 else float(argv[3])
    word_indices = read_index_dict(argv[1])
    error_mat = np.load(argv[2])

    make_error_analysis(error_mat, word_indices)
