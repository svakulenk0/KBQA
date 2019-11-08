#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Nov 8, 2019

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Functions to load pre-trained models for KBQA
'''

from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, Bidirectional, TimeDistributed
from keras.optimizers import *


# define bi-LSTM model architecture (loose embeddings layer to do on-the-fly embedding at inference time)
def build_qt_inference_model(model_settings):
    # architecture
    _input = Input(shape=(model_settings['max_len'], model_settings['emb_dim']), name='input')
    model = Bidirectional(LSTM(units=100, return_sequences=True, dropout=0.5,
                               recurrent_dropout=0.5), name='bilstm1')(_input)  # biLSTM
    model = Bidirectional(LSTM(units=100, return_sequences=False, dropout=0.5,
                               recurrent_dropout=0.5), name='bilstm2')(model)  # 2nd biLSTM
    _output = Dense(model_settings['n_tags'], activation='softmax', name='output')(model)  # a dense layer
    model = Model(_input, _output)
    model.compile(optimizer=Nadam(clipnorm=1), loss='categorical_crossentropy', metrics=['accuracy']) 
    model.summary()
    return model

# load pre-trained EP spans parsing network
modelname = '2hops-types'

from keras_contrib.layers import CRF
from keras_contrib import losses, metrics


# prediction time span generator
def collect_mentions(words, y_p, tag_ind):
    e_span, e_spans = [], []
    for w, pred in zip(words, y_p):
        if pred == tag_ind:
            e_span.append(w)
        elif e_span:
            e_spans.append(" ".join(e_span))
            e_span = []
    # add last span
    if e_span:
        e_spans.append(" ".join(e_span))
        e_span = []
    # remove duplicates
    return list(set(e_spans))


# define bi-LSTM model architecture (loose embeddings layer to do on-the-fly embedding at inference time)
def build_ep_inference_model(model_settings):
    # architecture
    input = Input(shape=(model_settings['max_len'], model_settings['emb_dim']), name='input')
    model = Bidirectional(LSTM(units=100, return_sequences=True), name='bilstm1')(input)  # biLSTM
    model = Bidirectional(LSTM(units=100, return_sequences=True), name='bilstm2')(model)  # 2nd biLSTM
    model = TimeDistributed(Dense(model_settings['n_tags'], activation=None), name='td')(model)  # a dense layer
    crf = CRF(model_settings['n_tags'], name='crf')  # CRF layer
    out = crf(model)  # output
    model = Model(input, out)
    model.compile(optimizer=Nadam(lr=0.01, clipnorm=1), loss=losses.crf_loss, metrics=[metrics.crf_accuracy]) 
    model.summary()
    return model


# functions for entity linking and relation detection
def entity_linking(e_spans, verbose=False, cutoff=500, threshold=0): 
    guessed_ids = []
    for span in e_spans:
        span_ids = e_index.label_scores(span, top=cutoff, threshold=threshold, verbose=verbose, scale=0.3, max_degree=50000)
        guessed_ids.append(span_ids)
    return guessed_ids


def relation_detection(p_spans, verbose=False, cutoff=500, threshold=0.0): 
    guessed_ids = []
    for span in p_spans:
        span_ids = {}
        guessed_labels = []
        if span in p_vectors:
            guessed_labels.append([span, 1])
        for p, score in p_vectors.most_similar(span, topn=cutoff):
            if score >= threshold:
                guessed_labels.append([p, score])
        for label, score in guessed_labels:
            for match in p_index.look_up_by_label(label):
                _id = match['_source']['id']
                span_ids[_id] = score
                if verbose:
                    uri = match['_source']['uri']
                    print(uri)
                    print(score)
        guessed_ids.append(span_ids)
    return guessed_ids

import re, string

def preprocess_span(span):
    entity_label = " ".join(re.sub('([a-z])([A-Z])', r'\1 \2', span).split())
    words = entity_label.split('_')
    unique_words = []
    for word in words:
        # strip punctuation
        word = "".join([c for c in word if c not in string.punctuation])
        if word:
            word = word.lower()
            if word not in unique_words:
                unique_words.append(word)
    return " ".join(unique_words)

