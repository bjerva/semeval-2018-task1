#!/usr/bin/env python

'''
SemEval-2018 Task 1

Run with python -O to skip debugging printouts.
Run with python -u to make sure slurm logs are written instantly.
'''

# Random seeds
import numpy as np
import random
random.seed(1337)
np.random.seed(1337)  # Freeze seeds for reproducibility

# Keras
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import LSTM, GRU, Input, add, concatenate, BatchNormalization
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Lambda
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras import backend as K
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ProgbarLogger

# Standard
import os
import argparse
from collections import defaultdict

# Implementation-specific
import utils
import data_utils
from analysis import write_confusion_matrix, prepare_error_analysis
from config import *

from scipy.stats import pearsonr


def build_model():
    '''
    Build a Keras model with the functional API
    '''

    bn_mode = 1
    if args.chars:
        char_input = Input(shape=(args.max_word_len, ), dtype='int32', name='char_input')

        x = Embedding(char_vocab_size, args.char_embedding_dim, input_length=args.max_word_len)(char_input)
        x = Reshape((args.max_word_len, args.char_embedding_dim))(x)

        prev_x = x
        for rnet_idx in range(args.resnet):
            if args.bn:
                x = BatchNormalization(mode=bn_mode)(x)
            x = Activation('relu')(x)
            if args.dropout:
                x = Dropout(args.dropout)(x)

            x = Convolution1D(args.char_embedding_dim, 8, activation='relu', padding='same')(x)

            if args.bn:
                x = BatchNormalization(mode=bn_mode)(x)
            x = Activation('relu')(x)
            if args.dropout:
                x = Dropout(args.dropout)(x)

            x = Convolution1D(args.char_embedding_dim, 4, activation='relu', padding='same')(x)

            x = add([prev_x, x])
            x = MaxPooling1D(pool_size=2, padding='same')(x)
            prev_x = x

        if args.bn:
            x = BatchNormalization(mode=bn_mode)(x)

        x = Activation('relu')(x)

        feature_size = args.char_embedding_dim
        char_embedding = Reshape((int(args.max_word_len / (2 ** args.resnet)), int(feature_size)))(x)

    if args.words:
        word_input = Input(shape=(args.max_sent_len, ), dtype='int32', name='word_input')
        if not args.ignore_embeddings:
            word_embedding = Embedding(vocab_size, word_embedding_dim, input_length=args.max_sent_len, weights=[embedding_weights], trainable=(args.freeze))(word_input)
        else:
            word_embedding = Embedding(vocab_size, word_embedding_dim, input_length=args.max_sent_len)(word_input)

        l = GRU(units=int(args.rnn_dim), return_sequences=False, dropout=args.dropout, input_shape=(args.max_sent_len, word_embedding_dim), activation='relu')(word_embedding)
        #l = GRU(units=int(args.rnn_dim)/2, return_sequences=False, dropout=args.dropout, input_shape=(args.max_sent_len, word_embedding_dim), activation='relu')(l)
        r = GRU(units=int(args.rnn_dim), return_sequences=False, go_backwards=True, dropout=args.dropout, input_shape=(args.max_sent_len, word_embedding_dim), activation='relu')(word_embedding)
        #r = GRU(units=int(args.rnn_dim)/2, return_sequences=False, go_backwards=True, dropout=args.dropout, input_shape=(args.max_sent_len, word_embedding_dim), activation='relu')(r)

        word_x = concatenate([l, r])
        if args.bn:
            word_x = BatchNormalization(mode=bn_mode)(word_x)

        if args.dropout:
            word_x = Dropout(args.dropout)(word_x)

    if args.chars:
        embedding = char_embedding

        if args.bn:
            embedding = BatchNormalization(mode=bn_mode)(embedding)

        if args.rnn:
            # Bidirectional GRU
            l = GRU(units=int(args.rnn_dim), return_sequences=False, dropout=args.dropout, input_shape=(args.max_sent_len, args.char_embedding_dim), activation='relu')(embedding)
            #l = GRU(units=int(args.rnn_dim)/2, return_sequences=False, dropout=args.dropout, input_shape=(args.max_sent_len, args.char_embedding_dim), activation='relu')(l)

            r = GRU(units=int(args.rnn_dim), return_sequences=False, go_backwards=True, dropout=args.dropout, input_shape=(args.max_sent_len, args.char_embedding_dim), activation='relu')(embedding)
            #r = GRU(units=int(args.rnn_dim)/2, return_sequences=False, go_backwards=True, dropout=args.dropout, input_shape=(args.max_sent_len, args.char_embedding_dim), activation='relu')(r)

            x = concatenate([l, r])
            if args.bn:
                x = BatchNormalization(mode=bn_mode)(x)
        else:

            x = Convolution1D(args.char_embedding_dim, 8, activation='relu', border_mode='same')(embedding)
            #
            x = MaxPooling1D(pool_length=8, border_mode='same')(x)

            x = Convolution1D(args.char_embedding_dim, 4, activation='relu', border_mode='same')(x)
            #
            x = MaxPooling1D(pool_length=8, border_mode='same')(x)

            x = Reshape((args.char_embedding_dim * 2, ))(x)
            x = Dense(args.char_embedding_dim, activation='relu')(x)

    if args.chars and args.words:
        x = concatenate([x, word_x])
        x = Dense(word_embedding_dim * 2, activation='relu')(x)
    elif args.words:
        x = word_x

    # Output layer
    main_output = Dense(1, activation='linear', name='main_output')(x)

    if args.chars and args.words:
        model_input = [word_input, char_input]
    elif args.chars:
        model_input = [char_input, ]
    elif args.words:
        model_input = [word_input, ]

    model_output = [main_output, ]

    model = Model(inputs=model_input, outputs=model_output)

    return model

def evaluate(model):
    '''
    TODO: Document
    '''
    print('Dev set results:')

    classes = model.predict(X_dev, batch_size=args.bsize)
    dev_classes, dev_accuracy, dev_tags = calculate_accuracy(model, y_dev, classes, os.path.basename(args.dev[0]))

    if args.bookkeeping:
        with open(experiment_dir+'/dev_acc.txt', 'w') as out_f:
            out_f.write('Dev accuracy: {0}\n'.format(dev_accuracy*100))

        save_outputs(dev_tags, X_dev_words, os.path.basename(args.dev[0]))

    if args.test:
        print('Test set')
        classes = model.predict(X_test, batch_size=args.bsize)

        test_classes, test_accuracy, test_tags = calculate_accuracy(model, y_test, classes, os.path.basename(args.test[0]))

        if args.bookkeeping:
            with open(experiment_dir+'/test_acc.txt', 'w') as out_f:
                out_f.write('Test accuracy: {0}\n'.format(test_accuracy*100))

            save_outputs(test_tags, X_test_words, os.path.basename(args.test[0]))

    print('Sanity check on train set:')
    classes = model.predict(X_train, batch_size=args.bsize)

    calculate_accuracy(model, y_train, classes, os.path.basename(args.train[0]))

    return dev_classes

def calculate_accuracy(model, y, classes, fname):
    '''
    TODO: Document
    '''
    
    sent_tags = []
    diff = 0
    for idx, sentence in enumerate(y):
        gold_tag = sentence
        pred_tag = classes[idx]
        diff += abs(gold_tag - pred_tag)

        indices = [idx]
        #print(str(gold_tag) + "    " + str(pred_tag))
        #print(str(gold_tag) + " " + str(pred_tag))
        sent_tags.append((indices, gold_tag, pred_tag))
    
    #print(np.asarray(y))
    #print("her kommer classes")
    #print(classes[:,0])
    accuracy = pearsonr(np.asarray(y), classes[:,0])
    print('Accuracy:', accuracy)
    sent_tags.append((indices, gold_tag, pred_tag))

    return classes, accuracy, sent_tags


def save_outputs(tags, X_words, fname):
    with open(experiment_dir+'/{0}_tag2id.txt'.format(fname), 'r') as in_f:
        id_to_tag = dict((line.strip().split()[::-1] for line in in_f))

    with open(experiment_dir+'/{0}_outputs.txt'.format(fname), 'w') as out_f:
        for sentence in tags:
            out_f.write(u'{0}\n'.format(id_to_tag[str(sentence[2])]))

def save_run_information():
    '''
    FIXME: Several things not implemented.
    '''
    try:
        from keras.utils.visualize_util import plot
        plot(model, to_file=experiment_dir+'/model.png', show_shapes=True)
    except:
        print('Could not save model plot...')

    try:
        model.save_weights(experiment_dir+'/weights.h5')
    except ImportError:
        print('Could not save weights...')

    json_string = model.to_json()
    with open(experiment_dir+'/architecture.json', 'w') as out_f:
        out_f.write(json_string)

    try:
        write_confusion_matrix(y_dev, dev_classes, nb_classes)
    except:
        print('Conf matrix not written')

    try:
        prepare_error_analysis(X_dev, y_dev, dev_classes, vocab_size)
    except:
        print('Error analysis not written')

if __name__ == '__main__':
    print("use chars?", args.chars)

    if args.embeddings:
        if __debug__: print('Loading embeddings...')
        word_vectors, index_dict, word_embedding_dim = utils.read_word_embeddings(args.embeddings)
        #print(next(key for key, value in index_dict.items() if value == 7241))
        if __debug__: print('Embeddings for {} words loaded'.format(len(word_vectors)))
    else:
        word_embedding_dim = args.word_embedding_dim   ### TODO: if no embeddings given, no index_dict!
        index_dict = {} # HACK: Empty dict will do for now
        word_vectors = None

    if __debug__: print('Loading data...')

    # Word data must be read even if word features aren't used
    (X_train_words, y_train), (X_dev_words, y_dev), (X_test_words, y_test), word_vectors = data_utils.read_word_data(args.train[0], args.dev[0], args.test[0], index_dict, word_vectors, args.max_sent_len)
    nb_classes = 1
    #print(nb_classes)
    #print(len(y_train[0]))

    if args.words:
        if args.ignore_embeddings or not args.embeddings:
            vocab_size = 1 + max(np.max(X_train_words), max(np.max(X_dev_words), np.max(X_test_words)))
        else:
            vocab_size = len(index_dict)
            #print(vocab_size)
            embedding_weights = np.zeros((vocab_size, word_embedding_dim))
            for word, index in index_dict.items():
                if __debug__ and word not in word_vectors:
                    print('word not in vectors', word)
                    continue
                try:
                    embedding_weights[index,:] = word_vectors[word]
                    #print(word_vectors[word])
                except ValueError:
                    embedding_weights[index,:] = np.zeros(300)
                #print(word + " " + str(index_dict[word]))

    if args.chars:
        if __debug__:
            print('Loading {0} features...'.format('byte' if args.bytes else 'char'))

        char_to_id = defaultdict(lambda: len(char_to_id))
        for dummy_char in (SENT_PAD, SENT_START, SENT_END, UNKNOWN):
            char_to_id[dummy_char]

        X_train_chars, X_dev_chars, X_test_chars = data_utils.read_char_data(args.train[0], args.dev[0], args.test[0], char_to_id, args.max_sent_len, args.max_word_len)

        char_vocab_size = len(char_to_id)
        if __debug__:
            print('{0} char ids'.format(char_vocab_size))

    if __debug__: print('Building model...')

    if args.chars and args.words:
        X_train = [X_train_words, X_train_chars]
        X_dev = [X_dev_words, X_dev_chars]
        X_test = [X_test_words, X_test_chars]

    elif args.chars:
        X_train = [X_train_chars, ]
        X_dev = [X_dev_chars, ]
        X_test = [X_test_chars, ]
    elif args.words:
        X_train = [X_train_words, ]
        X_dev = [X_dev_words, ]
        X_test = [X_test_words, ]

    model_outputs = [y_train, ]
    model_losses = ['mean_squared_error', ]
    model_loss_weights = [1.0, ]

    def mean_pred(y_true, y_pred):
        return K.mean(abs(y_true-y_pred))

    model_metrics = [mean_pred, ]


    model = build_model()

    model.compile(optimizer='adam',
              loss=model_losses,
              loss_weights=model_loss_weights,
              metrics=model_metrics)

    model.summary()

    if __debug__: print('Fitting...')

    callbacks = [ProgbarLogger()]

    if args.early_stopping:
        callbacks.append(EarlyStopping(monitor='val_loss', patience=5))

    model.fit(X_train, y_train,
                  validation_data=(X_dev, y_dev),
                  epochs=args.epochs,
                  batch_size=args.bsize,
                  callbacks=callbacks,
                  verbose=args.verbose)

    if __debug__:
        print(args)
        print('Evaluating...')

    dev_classes = evaluate(model)

    if args.bookkeeping:
        save_run_information()

    print('Completed: {0}'.format(experiment_tag))
