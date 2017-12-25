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
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ProgbarLogger, TerminateOnNaN
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
from config import *

#sys.path.append('../')
#import kode.testkode.eval as ev
from scipy.stats import pearsonr
from sklearn.model_selection import KFold

np.set_printoptions(suppress=True)

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
            word_embedding = Embedding(vocab_size, word_embedding_dim, input_length=args.max_sent_len, weights=[embedding_weights], trainable=(not args.freeze), name='word_embedding')(word_input)
        else:
            word_embedding = Embedding(vocab_size, word_embedding_dim, input_length=args.max_sent_len, name='word_embedding')(word_input)

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

emdict = {'anger' : 0, 'fear' : 1, 'joy' : 2, 'sadness' : 3}
revdict = {1701 : 'anger', 2252 : 'fear', 1616 : 'joy', 1533 : 'sadness'}

def calculate_reg(gold, preds):
    pearson = pearsonr(gold, preds)[0]

    gold_high = []
    pred_high = []

    for i, value in enumerate(gold):
        if value > 0.5:
            gold_high.append(gold[i])
            pred_high.append(preds[i])

    pearson_high = pearsonr(gold_high, pred_high)[0]

    return np.round(pearson, decimals=3), np.round(pearson_high, decimals=3)

def calculate_class(preds, gold):
    for i, augmented in enumerate(gold):
        if sum(augmented) < 0:
            gold = np.delete(gold, i, axis=0)
            preds = np.delete(preds, i, axis=0)
    micro_accuracy = []

    actual_emotion_micro = [0]*12
    correct_emotion_micro = [0]*12
    assigned_emotion_micro = [0]*12

    p_micro = [0]*12
    r_micro = [0]*12
    f_micro = [0]*12
    avg_f_micro = 0

    p_macro = 0
    r_macro = 0
    avg_f_macro = 0


    for i, labels in enumerate(gold):
        
        #Convert to class representation
        gold_labels = np.where(labels == 1)[0]
        pred_labels = np.where(preds[i] == 1)[0]

        #Neutral emotions, all 11 emotions 0
        if len(gold_labels) == 0:
            gold_labels = np.append(gold_labels, 11)
        if len(pred_labels) == 0:
            pred_labels = np.append(pred_labels, 11)
        
        for value_gold in gold_labels:
            actual_emotion_micro[value_gold] += 1
            if value_gold in pred_labels:
                correct_emotion_micro[value_gold] += 1

        for value_pred in pred_labels:
            assigned_emotion_micro[value_pred] += 1
        
        intersection = len(np.intersect1d(gold_labels, pred_labels))
        union = len(np.union1d(gold_labels, pred_labels))
        micro_accuracy.append(intersection/union)

    macro_accuracy = sum(micro_accuracy)/len(gold)

    for i in range(12):
        try:
            p_micro[i] = correct_emotion_micro[i]/assigned_emotion_micro[i]
        except ZeroDivisionError:
            p_micro[i] = 0
        
        try:
            r_micro[i] = correct_emotion_micro[i]/actual_emotion_micro[i]
        except ZeroDivisionError:
            r_micro[i] = 0

        try:
            f_micro[i] = 2*p_micro[i]*r_micro[i]/(p_micro[i]+r_micro[i])
        except ZeroDivisionError:
            f_micro[i] = 0
    
    p_micro = np.round(p_micro, decimals=3)
    r_micro = np.round(r_micro, decimals=3)
    f_micro = np.round(f_micro, decimals=3)

    avg_f_micro = sum(f_micro)/len(f_micro)

    p_macro = sum(correct_emotion_micro)/sum(assigned_emotion_micro)
    r_macro = sum(correct_emotion_micro)/sum(actual_emotion_micro)
    try:
        avg_f_macro = 2*p_macro*r_macro/(p_macro+r_macro)
    except ZeroDivisionError:
        avg_f_macro = 0
    
    avg_f_macro = round(avg_f_macro, 3)
    avg_f_micro = round(avg_f_micro, 3)
    
    return macro_accuracy, p_micro, r_micro, f_micro, avg_f_micro, p_macro, r_macro, avg_f_macro
    

def ev(train_preds, train_labels, dev_preds, dev_labels, test_preds, test_labels):
    helper_string = ''
    if len(train_preds) == 4:
        helper_string += ('Sanity check:\n')
        pearson_avg_train = []
        for i, gold in enumerate(train_labels):
            pearson, pearson_high = calculate_reg(gold, train_preds[i])
            helper_string += ("Pearson for train tweets, {0}: {1}\n".format(revdict[i], pearson))
            helper_string += ("Pearson for > 0.5 train tweets, {0}: {1}\n".format(revdict[i], pearson_high))
            pearson_avg_train.append(pearson)
        helper_string += ("Average Pearson for train tweets: {0:.3f}\n".format(sum(pearson_avg_train)/len(pearson_avg_train)))

        helper_string += ('Pearson for dev set:\n')
        pearson_avg_dev = []
        for i, gold in enumerate(dev_labels):
            pearson, pearson_high = calculate_reg(gold, dev_preds[i])
            helper_string += ("Pearson for dev tweets, {0}: {1}\n".format(revdict[i], pearson))
            helper_string += ("Pearson for > 0.5 dev tweets, {0}: {1}\n".format(revdict[i], pearson_high))
            pearson_avg_dev.append(pearson)
        helper_string += ("Average Pearson for dev tweets: {0:.3f}\n".format(sum(pearson_avg_dev)/len(pearson_avg_dev)))

        helper_string += ('Pearson for test set:\n')
        pearson_avg_test = []
        for i, gold in enumerate(test_labels):
            pearson, pearson_high = calculate_reg(gold, test_preds[i])
            helper_string += ("Pearson for test tweets, {0}: {1}\n".format(revdict[i], pearson))
            helper_string += ("Pearson for > 0.5 test tweets, {0}: {1}\n".format(revdict[i], pearson_high))
            pearson_avg_test.append(pearson)
        helper_string += ("Average Pearson for test tweets: {0:.3f}\n".format(sum(pearson_avg_test)/len(pearson_avg_test)))

    elif len(train_preds) > 12:
        helper_string += ('Sanity check:\n')
        pearson, pearson_high = calculate_reg(train_labels, train_preds)
        helper_string += ("Pearson for train tweets, {0}: {1}\n".format(revdict[len(train_preds)], pearson))
        helper_string += ("Pearson for > 0.5 train tweets, {0}: {1}\n".format(revdict[len(train_preds)], pearson_high))
        helper_string += ('Pearson for dev set:\n')
        pearson, pearson_high = calculate_reg(dev_labels, dev_preds)
        helper_string += ("Pearson for dev tweets, {0}: {1}\n".format(revdict[len(train_preds)], pearson))
        helper_string += ("Pearson for > 0.5 dev tweets, {0}: {1}\n".format(revdict[len(train_preds)], pearson_high))
        helper_string += ('Pearson for test set:\n')
        pearson, pearson_high = calculate_reg(test_labels, test_preds)
        helper_string += ("Pearson for test tweets, {0}: {1}\n".format(revdict[len(train_preds)], pearson))
        helper_string += ("Pearson for > 0.5 test tweets, {0}: {1}\n".format(revdict[len(train_preds)], pearson_high))
    else:
        helper_string += ('Sanity check:\n')
        macro_accuracy, p_micro, r_micro, f_micro, avg_f_micro, p_macro, r_macro, avg_f_macro = calculate_class(train_preds, train_labels)
        helper_string += ("Global accuracy for train tweets: {0:.3f}\n".format(macro_accuracy))
        helper_string += ("F-micro for emotion classes:\n")
        helper_string += str(f_micro)
        helper_string += '\n'
        helper_string += ("and averaged: " + str(avg_f_micro)+'\n')

        helper_string += ('Accuracy for dev set:\n')
        macro_accuracy, p_micro, r_micro, f_micro, avg_f_micro, p_macro, r_macro, avg_f_macro = calculate_class(dev_preds, dev_labels)
        helper_string += ("Global accuracy for dev tweets: {0:.3f}\n".format(macro_accuracy))
        helper_string += ("F-micro for emotion classes:\n")
        helper_string += str(f_micro)
        helper_string += '\n'
        helper_string += ("and averaged: " + str(avg_f_micro)+'\n')

        helper_string += ('Accuracy for test set:\n')
        macro_accuracy, p_micro, r_micro, f_micro, avg_f_micro, p_macro, r_macro, avg_f_macro = calculate_class(test_preds, test_labels)
        helper_string += ("Global accuracy for test tweets: {0:.3f}\n".format(macro_accuracy))
        helper_string += ("F-micro for emotion classes:\n")
        helper_string += str(f_micro)
        helper_string += '\n'
        helper_string += ("and averaged: " + str(avg_f_micro)+'\n')
    return helper_string

def evaluate(model):
    '''
    TODO: Document
    '''
    train_preds = model.predict(X_train, batch_size=args.bsize, verbose=1)
    dev_preds = model.predict(X_dev, batch_size=args.bsize, verbose=1)
    test_preds = model.predict(X_test, batch_size=args.bsize, verbose=1)
    #save_outputs(y_train_reg, y_train_class, train_preds)

    ''' sent_ids = printPredsToFileReg(args.dev[0], './preds/sub/EI-reg_en_anger_pred.txt', dev_preds[:,0][:dev_lengths[0]])
    sent_ids.extend(printPredsToFileReg(args.dev[1], './preds/sub/EI-reg_en_fear_pred.txt', dev_preds[:,0][dev_lengths[0]:dev_lengths[1]]))
    sent_ids.extend(printPredsToFileReg(args.dev[2], './preds/sub/EI-reg_en_joy_pred.txt', dev_preds[:,0][dev_lengths[1]:dev_lengths[2]]))
    sent_ids.extend(printPredsToFileReg(args.dev[3], './preds/sub/EI-reg_en_sadness_pred.txt', dev_preds[:,0][dev_lengths[2]:dev_lengths[3]]))

    printPredsToFileClass(args.aux[1], './preds/sub/E-C_en_pred.txt', dev_preds[:,1:], sent_ids) '''
    if args.solo:
        helper_string = ev([train_preds[:,0][:train_lengths[0]],train_preds[:,0][train_lengths[0]:train_lengths[1]],
                    train_preds[:,0][train_lengths[1]:train_lengths[2]],train_preds[:,0][train_lengths[2]:train_lengths[3]]],
                    [y_train[:,0][:train_lengths[0]],y_train[:,0][train_lengths[0]:train_lengths[1]],
                    y_train[:,0][train_lengths[1]:train_lengths[2]],y_train[:,0][train_lengths[2]:train_lengths[3]]],

                    [dev_preds[:,0][:dev_lengths[0]],dev_preds[:,0][dev_lengths[0]:dev_lengths[1]],
                    dev_preds[:,0][dev_lengths[1]:dev_lengths[2]],dev_preds[:,0][dev_lengths[2]:dev_lengths[3]]],
                    [y_dev[:,0][:dev_lengths[0]],y_dev[:,0][dev_lengths[0]:dev_lengths[1]],
                    y_dev[:,0][dev_lengths[1]:dev_lengths[2]],y_dev[:,0][dev_lengths[2]:dev_lengths[3]]],

                    [test_preds[:,0][:test_lengths[0]],test_preds[:,0][test_lengths[0]:test_lengths[1]],
                    test_preds[:,0][test_lengths[1]:test_lengths[2]],test_preds[:,0][test_lengths[2]:test_lengths[3]]],
                    [y_test[:,0][:test_lengths[0]],y_test[:,0][test_lengths[0]:test_lengths[1]],
                    y_test[:,0][test_lengths[1]:test_lengths[2]],y_test[:,0][test_lengths[2]:test_lengths[3]]])
    else:
        helper_string = ev(train_preds[:,0], y_train[:,0], dev_preds[:,0], y_dev[:,0], test_preds[:,0], y_test[:,0])
    with open("./preds/{0}.txt".format(experiment_tag),'w') as f:
        f.write(helper_string)

def pred_statistics(fname):
    data = np.loadtxt(fname)

    anger_var = (np.var(data[:train_lengths[0],0]), np.var(data[:train_lengths[0],12]))
    anger_mean = (np.mean(data[:train_lengths[0],0]), np.mean(data[:train_lengths[0],12]))
    anger_sqrl = (data[:train_lengths[0],0] - data[:train_lengths[0],12])**2
    fear_var = (np.var(data[train_lengths[0]:train_lengths[1],0]), np.var(data[train_lengths[0]:train_lengths[1],12]))
    joy_var = (np.var(data[train_lengths[1]:train_lengths[2],0]), np.var(data[train_lengths[1]:train_lengths[2],12]))
    sadness_var = (np.var(data[train_lengths[2]:train_lengths[3],0]), np.var(data[train_lengths[2]:train_lengths[3],12]))
    print('Gold anger variance: {:.6f} \nPred anger variance: {:.6f}'.format(anger_var[0], anger_var[1]))
    print('Gold anger mean: {:.6f} \nPred anger mean: {:.6f}'.format(anger_mean[0], anger_mean[1]))
    #print(np.where(anger_sqrl > np.mean(anger_sqrl)))

def printPredsToFileReg(infile, outfile, res, infileenc="utf-8"):
    outf = open(outfile, 'wu', encoding=infileenc)
    sent_ids = []
    with open(infile, encoding=infileenc, mode='r') as f:
        outf.write(f.readline())
        for i, line in enumerate(f):
            outl = line.strip("\n").split("\t")
            outl[3] = str(round(res[i],3))
            outf.write("\t".join(outl) + '\n')
            sent_ids.append(outl[0])
    outf.close()
    return sent_ids

def printPredsToFileClass(infile, outfile, res, sent_ids, infileenc="utf-8"):
    outf = open(outfile, 'w', encoding=infileenc)
    with open(infile, encoding=infileenc, mode='r') as f:
        outf.write(f.readline())
        for i, line in enumerate(f):
            outl = line.strip("\n").split("\t")
            idx = sent_ids.index(outl[0])
            for j in range(len(outl[2:])):
                outl[j+2] = str(int(res[idx][j]))
            outf.write("\t".join(outl) + '\n')
    outf.close()

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
    
    (X_train_word, y_train), (X_dev_word, y_dev), (X_test_word, y_test), word_vectors, index_dict, train_lengths, dev_lengths, test_lengths = data_utils.read_word_data(args.train, args.dev, args.test, args.aux, index_dict, word_vectors, args.max_sent_len)
    X_train_word = np.asarray(X_train_word)
    X_dev_word = np.asarray(X_dev_word)
    X_test_word = np.asarray(X_test_word)

    y_train = np.asarray(y_train)

    y_dev = np.asarray(y_dev)
    
    y_test = np.asarray(y_test)

    if args.words:
        if args.ignore_embeddings or not args.embeddings:
            vocab_size = 1 + max(np.max(X_train_word), max(np.max(X_dev_word), np.max(X_test_word)))
        else:
            vocab_size = len(index_dict)

            embedding_weights = np.zeros((vocab_size, word_embedding_dim))
            for word, index in index_dict.items():
                if __debug__ and word not in word_vectors:
                    print('word not in vectors', word)
                    cntr += 1
                    continue
                embedding_weights[index,:] = word_vectors[word]

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


    model_outputs = [y_train]

    model_losses = {'main_output' : 'mean_squared_error'}
    model_loss_weights = [1]

    def mean_pred(y_true, y_pred):
        return K.mean(K.abs(y_true - y_pred))

    model_metrics = {'main_output' : mean_pred} 

    if args.reuse:
        print('Loading model...')
        model = load_model(args.reuse, custom_objects={'customAuxLoss': customAuxLoss, 'mean_pred': mean_pred, 'customAuxMetric' : customAuxMetric})
        evaluate(model)

        if args.plot:
            plot_model(model, to_file='model.png')

        if args.save_word_weights:
            print('HUSK AT INDEX DICT IKKE ER DET SAMME SOM DET MODELEN BLEV TRAENET PÅ NOEDVENDIGVIS!')
            layer = model.get_layer(name='word_embedding').get_weights()
            words_used = list(index_dict.keys())            
            with open('trained_embeddings.txt', 'w') as f:
                bad_chars = ['[', ']', '\n',]
                for index, word in enumerate(words_used):
                    weight = np.array2string(layer[0][index], precision=20)
                    for chars in bad_chars:
                        weight = weight.replace(chars, '')
                    weight = weight.replace('  ', ' ')
                    f.write('{0} {1}\n'.format(word, weight))
        quit()
    else:
        model = build_model()

    optimizer = optimizers.Adam(lr=0.001, decay=1e-6, clipnorm=1)
    if args.nadam:
        optimizer = optimizers.Nadam(lr=0.001, decay=1e-6, clipnorm=1)
    
    model.compile(optimizer=optimizer,
        loss=model_losses,
        loss_weights=model_loss_weights,
        metrics=model_metrics)

    model.summary()
    
    if args.plot:
        plot_model(model, to_file='model.png')

    if __debug__: print('Fitting...')
    callbacks = [TensorBoard(log_dir='./logs/{0}'.format(experiment_tag)),
                TerminateOnNaN()]

    if args.early_stopping:
        callbacks.append(EarlyStopping(monitor='val_loss', patience=args.early_stopping))

    if args.kfold:
        kf = KFold(n_splits=args.kfold, shuffle=True)
        for train_index, test_index in kf.split(X_train_word):
            import ipdb; ipdb.set_trace()
            model.fit([X_train[0][train_index], X_train[1][train_index]], [y_train_reg[train_index], 
                            y_train_class[:,0][train_index], y_train_class[:,1][train_index], y_train_class[:,2][train_index],
                            y_train_class[:,3][train_index], y_train_class[:,4][train_index], y_train_class[:,5][train_index], 
                            y_train_class[:,6][train_index], y_train_class[:,7][train_index], y_train_class[:,8][train_index],
                            y_train_class[:,9][train_index], y_train_class[:,10][train_index]],
                        validation_data=([X_train[0][test_index], X_train[1][test_index]], [y_train_reg[test_index], 
                            y_train_class[:,0][test_index], y_train_class[:,1][test_index], y_train_class[:,2][test_index],
                            y_train_class[:,3][test_index], y_train_class[:,4][test_index], y_train_class[:,5][test_index], 
                            y_train_class[:,6][test_index], y_train_class[:,7][test_index], y_train_class[:,8][test_index],
                            y_train_class[:,9][test_index], y_train_class[:,10][test_index]]),
                        epochs=args.epochs,
                        shuffle=True,
                        batch_size=args.bsize,
                        callbacks=callbacks,
                        verbose=args.verbose)
    else:
        model.fit(X_train, model_outputs,
            validation_data=(X_dev, [y_dev,]),
                epochs=args.epochs,
                shuffle=True,
                batch_size=args.bsize,
                callbacks=callbacks,
                verbose=args.verbose)

    if __debug__:
        print(args)
        print('Evaluating...')
    
    evaluate(model)

    if args.save:
        model.save("models/{0}.h5".format(experiment_tag))

    if args.save_word_weights:
        print('Saving word embedding weights...')
        layer = model.get_layer(name='word_embedding').get_weights()
        words_used = list(index_dict.keys())
        
        #word_dictionary = {}
        #for i, word in enumerate(words_used):
        #    word_dictionary[word] = layer[0][i]
        
        with open('trained_embeddings.txt', 'w') as f:
            bad_chars = ['[', ']', '\n',]
            for index, word in enumerate(words_used):
                weight = np.array2string(layer[0][index], precision=20)
                for chars in bad_chars:
                    weight = weight.replace(chars, '')
                weight = weight.replace('  ', ' ')
                f.write('{0} {1}\n'.format(word, weight))

    print('Completed: {0}'.format(experiment_tag))
