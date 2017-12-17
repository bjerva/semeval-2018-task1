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
from keras.models import Model, load_model
from keras.layers.embeddings import Embedding
from keras.layers import LSTM, GRU, Input, add, concatenate, BatchNormalization
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Lambda, Masking
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras import backend as K
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ProgbarLogger
from keras import metrics
from keras import optimizers
from keras.utils import plot_model
import pydot
import tensorflow as tf

# Standard
import os
import sys
import argparse
from collections import defaultdict

# Implementation-specific
import utils
import data_utils
from analysis import write_confusion_matrix, prepare_error_analysis
from config import *

sys.path.append('../')
import kode.testkode.eval as ev
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
                x = BatchNormalization()(x)
            x = Activation('relu')(x)
            if args.dropout:
                x = Dropout(args.dropout)(x)

            x = Convolution1D(args.char_embedding_dim, 8, activation='relu', padding='same')(x)

            if args.bn:
                x = BatchNormalization()(x)
            x = Activation('relu')(x)
            if args.dropout:
                x = Dropout(args.dropout)(x)

            x = Convolution1D(args.char_embedding_dim, 4, activation='relu', padding='same')(x)

            x = add([prev_x, x])
            x = MaxPooling1D(pool_size=2, padding='same')(x)
            prev_x = x

        if args.bn:
            x = BatchNormalization()(x)

        x = Activation('linear')(x)

        feature_size = args.char_embedding_dim
        char_embedding = Reshape((int(args.max_word_len / (2 ** args.resnet)), int(feature_size)))(x)

    if args.words:
        word_input = Input(shape=(args.max_sent_len, ), dtype='int32', name='word_input')
        if not args.ignore_embeddings:
            word_embedding = Embedding(vocab_size, word_embedding_dim, input_length=args.max_sent_len, weights=[embedding_weights], trainable=(not args.freeze))(word_input)
        else:
            word_embedding = Embedding(vocab_size, word_embedding_dim, input_length=args.max_sent_len)(word_input)

        l = GRU(units=int(args.rnn_dim), return_sequences=False, dropout=args.dropout, input_shape=(args.max_sent_len, word_embedding_dim), activation='relu')(word_embedding)
        #l = GRU(units=int(args.rnn_dim)/2, return_sequences=False, dropout=args.dropout, input_shape=(args.max_sent_len, word_embedding_dim), activation='relu')(l)
        r = GRU(units=int(args.rnn_dim), return_sequences=False, go_backwards=True, dropout=args.dropout, input_shape=(args.max_sent_len, word_embedding_dim), activation='relu')(word_embedding)
        #r = GRU(units=int(args.rnn_dim)/2, return_sequences=False, go_backwards=True, dropout=args.dropout, input_shape=(args.max_sent_len, word_embedding_dim), activation='relu')(r)

        word_x = concatenate([l, r])
        if args.bn:
            word_x = BatchNormalization()(word_x)

        if args.dropout:
            word_x = Dropout(args.dropout)(word_x)

    if args.chars:
        embedding = char_embedding

        if args.bn:
            embedding = BatchNormalization()(embedding)

        if args.rnn:
            # Bidirectional GRU
            l = GRU(units=int(args.rnn_dim), return_sequences=False, dropout=args.dropout, input_shape=(args.max_sent_len, args.char_embedding_dim), activation='relu')(embedding)
            #l = GRU(units=int(args.rnn_dim)/2, return_sequences=False, dropout=args.dropout, input_shape=(args.max_sent_len, args.char_embedding_dim), activation='relu')(l)

            r = GRU(units=int(args.rnn_dim), return_sequences=False, go_backwards=True, dropout=args.dropout, input_shape=(args.max_sent_len, args.char_embedding_dim), activation='relu')(embedding)
            #r = GRU(units=int(args.rnn_dim)/2, return_sequences=False, go_backwards=True, dropout=args.dropout, input_shape=(args.max_sent_len, args.char_embedding_dim), activation='relu')(r)

            x = concatenate([l, r])
            if args.bn:
                x = BatchNormalization()(x)
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
    #aux_mask
    anger_output = Dense(1, activation='sigmoid', name='anger_output')(x)
    anticipation_output = Dense(1, activation='sigmoid', name='anticipation_output')(x)
    disgust_output = Dense(1, activation='sigmoid', name='disgust_output')(x)
    fear_output = Dense(1, activation='sigmoid', name='fear_output')(x)
    joy_output = Dense(1, activation='sigmoid', name='joy_output')(x)
    love_output = Dense(1, activation='sigmoid', name='love_output')(x)
    optimism_output = Dense(1, activation='sigmoid', name='optimism_output')(x)
    pessimism_output = Dense(1, activation='sigmoid', name='pessimism_output')(x)
    sadness_output = Dense(1, activation='sigmoid', name='sadness_output')(x)
    surprise_output = Dense(1, activation='sigmoid', name='surprise_output')(x)
    trust_output = Dense(1, activation='sigmoid', name='trust_output')(x)
    #pre_main = concatenate([x, aux_output])
    #main_mask
    main_output = Dense(1, activation='linear', name='main_output')(x)

    if args.chars and args.words:
        model_input = [word_input, char_input]
    elif args.chars:
        model_input = [char_input, ]
    elif args.words:
        model_input = [word_input, ]

    model_output = [main_output, anger_output, anticipation_output, disgust_output, fear_output,
                    joy_output, love_output, optimism_output, pessimism_output, sadness_output,
                    surprise_output, trust_output]

    model = Model(inputs=model_input, outputs=model_output)

    return model

def evaluate(model):
    '''
    TODO: Document
    '''
    train_preds = model.predict(X_train, batch_size=args.bsize, verbose=1)
    dev_preds = model.predict(X_dev, batch_size=args.bsize, verbose=1)
    test_preds = model.predict(X_test, batch_size=args.bsize, verbose=1)

    for i in range(11):
        train_preds[i+1] = np.round(train_preds[i+1])
        dev_preds[i+1] = np.round(dev_preds[i+1])
        test_preds[i+1] = np.round(test_preds[i+1])
        
    train_preds = np.c_[train_preds[0],train_preds[1],train_preds[2],train_preds[3],train_preds[4],train_preds[5],
                        train_preds[6],train_preds[7],train_preds[8],train_preds[9],train_preds[10],train_preds[11]]

    dev_preds = np.c_[dev_preds[0],dev_preds[1],dev_preds[2],dev_preds[3],dev_preds[4],dev_preds[5],
                        dev_preds[6],dev_preds[7],dev_preds[8],dev_preds[9],dev_preds[10],dev_preds[11]]

    test_preds = np.c_[test_preds[0],test_preds[1],test_preds[2],test_preds[3],test_preds[4],test_preds[5],
                        test_preds[6],test_preds[7],test_preds[8],test_preds[9],test_preds[10],test_preds[11]]
    import ipdb; ipdb.set_trace()
    save_outputs(y_train_reg, y_train_class, train_preds)
    ev.evaluate([train_preds[:,0][:train_lengths[0]],train_preds[:,0][train_lengths[0]:train_lengths[1]],
                train_preds[:,0][train_lengths[1]:train_lengths[2]],train_preds[:,0][train_lengths[2]:train_lengths[3]]],
                [y_train_reg[:train_lengths[0]],y_train_reg[train_lengths[0]:train_lengths[1]],
                y_train_reg[train_lengths[1]:train_lengths[2]],y_train_reg[train_lengths[2]:train_lengths[3]]],

                [dev_preds[:,0][:dev_lengths[0]],dev_preds[:,0][dev_lengths[0]:dev_lengths[1]],
                dev_preds[:,0][dev_lengths[1]:dev_lengths[2]],dev_preds[:,0][dev_lengths[2]:dev_lengths[3]]],
                [y_dev_reg[:dev_lengths[0]],y_dev_reg[dev_lengths[0]:dev_lengths[1]],
                y_dev_reg[dev_lengths[1]:dev_lengths[2]],y_dev_reg[dev_lengths[2]:dev_lengths[3]]],

                [test_preds[:,0][:test_lengths[0]],test_preds[:,0][test_lengths[0]:test_lengths[1]],
                test_preds[:,0][test_lengths[1]:test_lengths[2]],test_preds[:,0][test_lengths[2]:test_lengths[3]]],
                [y_test_reg[:test_lengths[0]],y_test_reg[test_lengths[0]:test_lengths[1]],
                y_test_reg[test_lengths[1]:test_lengths[2]],y_test_reg[test_lengths[2]:test_lengths[3]]])
    
    ev.evaluate(train_preds[:,1:],
                y_train_class,

                dev_preds[:,1:],
                y_dev_class,

                test_preds[:,1:],
                y_test_class)


def save_outputs(gold_reg, gold_class, preds):
    with open('preds.txt', mode='w') as f:
        for i in range(preds.shape[0]):
            gold = np.array2string(np.insert(gold_class[i],0,gold_reg[i])).replace('\n', '')
            pred = np.array2string(preds[i]).replace('\n','')
            f.write('{0}\t{1}\n'.format(gold,pred))
        

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

    #y_aux_class = np.empty([0, 11])
    
    (X_train_word, y_train), (X_dev_word, y_dev), (X_test_word, y_test), word_vectors, index_dict, train_lengths, dev_lengths, test_lengths = data_utils.read_word_data(args.train, args.dev, args.test, args.aux, index_dict, word_vectors, args.max_sent_len)
    X_train_word = np.asarray(X_train_word)
    X_dev_word = np.asarray(X_dev_word)
    X_test_word = np.asarray(X_test_word)

    y_train = np.asarray(y_train)
    y_train_reg = y_train[:,0]
    y_train_class = y_train[:,1:]

    y_dev = np.asarray(y_dev)
    y_dev_reg = y_dev[:,0]
    y_dev_class = y_dev[:,1:]
    
    y_test = np.asarray(y_test)
    y_test_reg = y_test[:,0]
    y_test_class = y_test[:,1:]

    #nb_classes = 1
    #print(nb_classes)
    #print(len(y_train[0]))

    if args.words:
        if args.ignore_embeddings or not args.embeddings:
            vocab_size = 1 + max(np.max(X_train_word), max(np.max(X_dev_word), np.max(X_test_word)))
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
                except ValueError: #tror ikke det er godt det her
                    embedding_weights[index,:] = np.zeros(word_embedding_dim)
                    #print(word + " " + str(index_dict[word]))

    if args.chars:
        if __debug__:
            print('Loading {0} features...'.format('byte' if args.bytes else 'char'))

        char_to_id = defaultdict(lambda: len(char_to_id))
        for dummy_char in (SENT_PAD, SENT_START, SENT_END, UNKNOWN):
            char_to_id[dummy_char]

        X_train_char, X_dev_char, X_test_char = data_utils.read_char_data(args.train, args.dev, args.test, char_to_id, args.max_sent_len, args.max_word_len)

        X_train_char = np.asarray(X_train_char)
        X_dev_char = np.asarray(X_dev_char)
        X_test_char = np.asarray(X_test_char)

        char_vocab_size = len(char_to_id)
        if __debug__:
            print('{0} char ids'.format(char_vocab_size))

    if __debug__: print('Building model...')

    if args.chars and args.words:
        X_train = [X_train_word, X_train_char]
        X_dev = [X_dev_word, X_dev_char]
        X_test = [X_test_word, X_test_char]
    
    elif args.chars:
        X_train = X_train_char
        X_dev = X_dev_char
        X_test = X_test_char

    elif args.words:
        X_train = X_train_word
        X_dev = X_dev_word
        X_test = X_test_word
    
    MASK = tf.convert_to_tensor([-1.0])

    def customMainLoss(y_true, y_pred):
        return K.mean(K.switch(K.equal(y_true, MASK), tf.multiply(y_true,0), K.square(y_true - y_pred)))


    def create_weighted_binary_crossentropy(zero_weight, one_weight):

        def weighted_binary_crossentropy(y_true, y_pred):

            # Original binary crossentropy (see losses.py):
            # K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

            # Calculate the binary crossentropy
            b_ce = K.binary_crossentropy(y_true, y_pred)

            # Apply the weights
            weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
            weighted_b_ce = weight_vector * b_ce

            # Return the mean error
            return weighted_b_ce

        return weighted_binary_crossentropy

    weighted_binary_crossentropy = create_weighted_binary_crossentropy(0.2, 0.8)

    def customAuxLoss(y_true, y_pred):
        return K.mean(K.switch(K.equal(y_true, MASK), tf.multiply(y_true,0), weighted_binary_crossentropy(y_true, y_pred)),axis=1)

    model_outputs = [y_train_reg, y_train_class[:,0], y_train_class[:,1], y_train_class[:,2], y_train_class[:,3],
                    y_train_class[:,4], y_train_class[:,5], y_train_class[:,6], y_train_class[:,7],
                    y_train_class[:,8], y_train_class[:,9], y_train_class[:,10]]

    model_losses = {'main_output' : 'mean_squared_error',
                    'anger_output' : customAuxLoss,
                    'anticipation_output' : customAuxLoss,
                    'disgust_output' : customAuxLoss,
                    'fear_output' : customAuxLoss,
                    'joy_output' : customAuxLoss,
                    'love_output' : customAuxLoss,
                    'optimism_output' : customAuxLoss,
                    'pessimism_output' : customAuxLoss,
                    'sadness_output' : customAuxLoss,
                    'surprise_output' : customAuxLoss,
                    'trust_output' : customAuxLoss} #, 'categorical_crossentropy'
    model_loss_weights = [0.45, 0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05]

    def customMainMetric(y_true, y_pred):
        return K.sum(K.switch(K.equal(y_true, MASK), tf.multiply(y_true,0), K.abs(y_true - y_pred)))

    def mean_pred(y_true, y_pred):
        return K.mean(K.abs(y_true - y_pred))

    def customAuxMetric(y_true, y_pred):
        return K.mean(K.switch(K.equal(y_true, MASK), tf.multiply(y_true,-1.0), K.cast(K.equal(y_true, K.round(y_pred)),dtype='float32')))

    model_metrics = {'main_output' : mean_pred,
                    'anger_output' : customAuxMetric,
                    'anticipation_output' : customAuxMetric,
                    'disgust_output' : customAuxMetric,
                    'fear_output' : customAuxMetric,
                    'joy_output' : customAuxMetric,
                    'love_output' : customAuxMetric,
                    'optimism_output' : customAuxMetric,
                    'pessimism_output' : customAuxMetric,
                    'sadness_output' : customAuxMetric,
                    'surprise_output' : customAuxMetric,
                    'trust_output' : customAuxMetric} #, 'categorical_crossentropy'

    if args.reuse:
        model = load_model(args.reuse, custom_objects={'customAuxLoss': customAuxLoss, 'mean_pred': mean_pred, 'customAuxMetric' : customAuxMetric})
        evaluate(model)
        quit()
    else:
        model = build_model()

    kf = KFold(n_splits=2, shuffle=True)

    adam = optimizers.Adam(lr=0.001, decay=1e-6, clipnorm=1)
    
    model.compile(optimizer=adam,
        loss=model_losses,
        loss_weights=model_loss_weights,
        metrics=model_metrics)

    model.summary()
    #plot_model(model, to_file='model.png')

    if __debug__: print('Fitting...')
    callbacks = [TensorBoard(log_dir='./logs/2ndovernite')]

    if args.early_stopping:
        callbacks.append(EarlyStopping(monitor='val_loss', patience=5))

    model.fit(X_train, model_outputs,
                validation_data=(X_dev, [y_dev_reg, y_dev_class[:,0], y_dev_class[:,1], y_dev_class[:,2], y_dev_class[:,3],
                    y_dev_class[:,4], y_dev_class[:,5], y_dev_class[:,6], y_dev_class[:,7],
                    y_dev_class[:,8], y_dev_class[:,9], y_dev_class[:,10]]),
                epochs=args.epochs,
                shuffle=True,
                batch_size=args.bsize,
                callbacks=callbacks,
                verbose=args.verbose)

    if __debug__:
        print(args)
        print('Evaluating...')

    evaluate(model)
    model.save("models/{0}.h5".format(experiment_tag))

    print('Completed: {0}'.format(experiment_tag))
