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
from keras import metrics

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
from sklearn.model_selection import KFold


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

        x = Activation('linear')(x)

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
    aux_output = Dense(emotions, activation='softmax', name='aux_output')(x)
    #pre_main = concatenate([x, aux_output])
    main_output = Dense(1, activation='linear', name='main_output')(x)

    if args.chars and args.words:
        model_input = [word_input, char_input]
    elif args.chars:
        model_input = [char_input, ]
    elif args.words:
        model_input = [word_input, ]

    model_output = [main_output, aux_output]

    model = Model(inputs=model_input, outputs=model_output)

    return model

def evaluate(model):
    '''
    TODO: Document
    '''
    print('Dev set results:')

    predictions = model.predict(X_dev, batch_size=args.bsize)
    calculate_accuracy(model, predictions, y_dev_labels, y_dev_class, os.path.basename(args.dev[0])) #BRUGER IKKE FNAME ARGUMENTET, SLET
    import ipdb; ipdb.set_trace()
    if args.test:
        print('Test set')
        predictions = model.predict(X_test, batch_size=args.bsize)
        calculate_accuracy(model, predictions, y_test_labels, y_test_class, os.path.basename(args.test[0]))

    print('Sanity check on train set:')
    predictions = model.predict(X_train, batch_size=args.bsize)
    calculate_accuracy(model, predictions, y_train_labels, y_train_class, os.path.basename(args.train[0]))

def calculate_accuracy(model, prediction, gold_reg, gold_class, fname):
    '''
    TODO: Document
    '''
    
    sent_tags = []
    corr, err = 0,0
    diff = 0
    for i, pred_reg in enumerate(prediction[0]):
        diff += abs(gold_reg[i] - pred_reg)

    for i, prob_class in enumerate(prediction[1]):
        pred_class = np.argmax(prob_class)
        if pred_class == np.argmax(gold_class[i]):
            corr += 1
        else:
            err += 1
    reg_accuracy = pearsonr(np.asarray(prediction[0]), np.reshape(gold_reg, (len(gold_reg),1)))
    print('Regression accuracy:', reg_accuracy)
    class_accuracy = corr/float(corr+err)
    print('Classification accuracy', class_accuracy)

    return prediction, reg_accuracy, class_accuracy


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

    if args.embeddings and not args.ignore_embeddings:
        if __debug__: print('Loading embeddings...')
        word_vectors, index_dict, word_embedding_dim = utils.read_word_embeddings(args.embeddings)
        #print(next(key for key, value in index_dict.items() if value == 7241))
        if __debug__: print('Embeddings for {} words loaded'.format(len(word_vectors)))
    else:
        word_embedding_dim = args.word_embedding_dim   ### TODO: if no embeddings given, no index_dict!
        index_dict = {}
        word_vectors = np.random.standard_normal(size=(args.nwords, args.word_embedding_dim))

    if __debug__: print('Loading data...')

    # Word data must be read even if word features aren't used
    emotions = len(args.train)

    X_train_words = []
    y_train_labels = []
    y_train_class = np.empty([0, emotions])

    X_dev_words = []
    y_dev_labels = []
    y_dev_class = np.empty([0, emotions])
    
    X_test_words = []
    y_test_labels = []
    y_test_class = np.empty([0, emotions])

    #y_aux_class = np.empty([0, 11])

    for index in range(emotions): #HIVER AUX DATA NED HVER ENESTE GANG DER BLIVER LÆST EN AF DE FIRE FILER NED!!!
        (X_train_word, y_train), (X_dev_word, y_dev), (X_test_word, y_test), (X_aux_words, y_aux), word_vectors, index_dict = data_utils.read_word_data(args.train[index], args.dev[index], args.test[index], args.aux, index_dict, word_vectors, args.max_sent_len)
        X_train_words.extend(X_train_word)
        y_train_labels.extend(y_train)

        helpme1 = np.zeros([X_train_word.shape[0], emotions])
        helpme1[:,index] = np.ones(X_train_word.shape[0])
        y_train_class = np.append(y_train_class, helpme1, axis=0)

        X_dev_words.extend(X_dev_word)
        y_dev_labels.extend(y_dev)

        helpme2 = np.zeros([X_dev_word.shape[0], emotions])
        helpme2[:,index] = np.ones(X_dev_word.shape[0])
        y_dev_class = np.append(y_dev_class, helpme2, axis=0)

        X_test_words.extend(X_test_word)
        y_test_labels.extend(y_test)

        helpme3 = np.zeros([X_test_word.shape[0], emotions])
        helpme3[:,index] = np.ones(X_test_word.shape[0])
        y_test_class = np.append(y_test_class, helpme3, axis=0)

    unique_words = len(set(np.concatenate(X_train_words)))

    X_train_words = np.concatenate(X_train_words).reshape(len(X_train_words), args.max_sent_len)
    X_dev_words = np.concatenate(X_dev_words).reshape(len(X_dev_words), args.max_sent_len)
    X_test_words = np.concatenate(X_test_words).reshape(len(X_test_words), args.max_sent_len)
    
    y_aux_class = np.asarray(y_aux)

    y_train_labels = np.asarray(y_train_labels)
    y_dev_labels = np.asarray(y_dev_labels)
    y_test_labels = np.asarray(y_test_labels)

    #nb_classes = 1
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
                    cntr += 1
                    continue
                try:
                    embedding_weights[index,:] = word_vectors[word]
                    #print(word_vectors[word])
                except ValueError:
                    embedding_weights[index,:] = np.zeros(word_embedding_dim)
                    #print(word + " " + str(index_dict[word]))

    if args.chars:
        if __debug__:
            print('Loading {0} features...'.format('byte' if args.bytes else 'char'))

        char_to_id = defaultdict(lambda: len(char_to_id))
        for dummy_char in (SENT_PAD, SENT_START, SENT_END, UNKNOWN):
            char_to_id[dummy_char]

        X_train_chars = np.empty([0, args.max_word_len])
        X_dev_chars = np.empty([0, args.max_word_len])
        X_test_chars = np.empty([0, args.max_word_len])
        for index in range(len(args.train)):
            X_train_char, X_dev_char, X_test_char, X_aux_chars = data_utils.read_char_data(args.train[index], args.dev[index], args.test[index], args.aux[0], char_to_id, args.max_sent_len, args.max_word_len)
            X_train_chars = np.append(X_train_chars, X_train_char, axis=0)
            X_dev_chars = np.append(X_dev_chars, X_dev_char, axis=0)
            X_test_chars = np.append(X_test_chars, X_test_char, axis=0)
        
        
        char_vocab_size = len(char_to_id)
        if __debug__:
            print('{0} char ids'.format(char_vocab_size))

    if __debug__: print('Building model...')

    if args.chars and args.words:
        X_train = [X_train_words, X_train_chars]
        X_dev = [X_dev_words, X_dev_chars]
        X_test = [X_test_words, X_test_chars]
    
    elif args.chars:
        X_train = X_train_chars
        X_dev = [X_dev_chars, ]
        X_test = [X_test_chars, ]
    elif args.words:
        X_train = X_train_words
        X_dev = [X_dev_words, ]
        X_test = [X_test_words, ]
    
    model_outputs = [y_train_labels, y_train_class]
    model_losses = ['mean_squared_error', 'categorical_crossentropy']
    model_loss_weights = [0.8, 0.2]

    def mean_pred(y_true, y_pred):
        return K.mean(abs(y_true-y_pred))

    model_metrics = {'main_output' : mean_pred,
                     'aux_output' : metrics.categorical_accuracy} #https://www.quora.com/How-does-Keras-calculate-accuracy


    model = build_model()

    kf = KFold(n_splits=9)

    model.compile(optimizer='adam',
        loss=model_losses,
        loss_weights=model_loss_weights,
        metrics=model_metrics)

    model.summary()

    for train_index, test_index in kf.split(X_train[0], y_train_labels):
        if __debug__: print('Fitting...')

        callbacks = [ProgbarLogger()]

        if args.early_stopping:
            callbacks.append(EarlyStopping(monitor='val_loss', patience=5))

        model.fit([X_train[0][train_index], X_train[1][train_index]], [y_train_labels[train_index], y_train_class[train_index]],
                    validation_data=([X_train[0][test_index], X_train[1][test_index]], [y_train_labels[test_index], y_train_class[test_index]]),
                    epochs=args.epochs,
                    batch_size=args.bsize,
                    callbacks=callbacks,
                    verbose=args.verbose)

    if __debug__:
        print(args)
        print('Evaluating...')

    evaluate(model)

    print('Completed: {0}'.format(experiment_tag))
