#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Feb 20, 2018

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Final evaluation script for comparison with the benchmark
'''

# setup
dataset_name = 'lcquad'
embeddings_choice = 'glove840B300d'

# connect to DB storing the dataset
from setup import Mongo_Connector, load_embeddings, IndexSearch
mongo = Mongo_Connector('kbqa', dataset_name)

# path to KG relations
from hdt import HDTDocument
# hdt_path = "/home/zola/Projects/hdt-cpp-molecules/libhdt/data/"
hdt_path = "/mnt/ssd/sv/"
hdt_file = 'dbpedia2016-04en.hdt'
namespace = "http://dbpedia.org/"

from collections import defaultdict
import numpy as np
import scipy.sparse as sp

# entity and predicate catalogs
e_index = IndexSearch('dbpedia201604e')
p_index = IndexSearch('dbpedia201604p')

from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, Bidirectional, TimeDistributed
from keras.optimizers import *
from keras.preprocessing.text import text_to_word_sequence

# load pre-trained Q type network
modelname = 'qtype'

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

# load model settings
import pickle as pkl
with open('%s_%s_%s.pkl'%(modelname, dataset_name, embeddings_choice), 'rb') as f:
    model_settings = pkl.load(f)

qt_model = build_qt_inference_model(model_settings)
# load weights
qt_model.load_weights('checkpoints/_'+modelname+'_weights.best.hdf5', by_name=True)

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

# load model settings
with open('%s_%s.pkl'%(dataset_name, embeddings_choice), 'rb') as f:
    ep_model_settings = pkl.load(f)

ep_model = build_ep_inference_model(ep_model_settings)
# load weights
ep_model.load_weights('model/'+modelname+'.h5', by_name=True)

# functions for entity linking and relation detection
def entity_linking(e_spans, cutoff=500, threshold=0): 
    guessed_ids = []
    for span in e_spans:
        span_ids = e_index.label_scores(span, top=cutoff, threshold=threshold, verbose=False, scale=0.3, max_degree=100000)
        guessed_ids.append(span_ids)
    return guessed_ids

def relation_detection(p_spans, cutoff=500, threshold=0.0): 
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

# load MP functions

def generate_adj_sp(adjacencies, n_entities, include_inverse):
    '''
    Build adjacency matrix
    '''
    adj_shape = (n_entities, n_entities)
    # colect all predicate matrices separately into a list
    sp_adjacencies = []

    for edges in adjacencies:
        # split subject (row) and object (col) node URIs
        n_edges = len(edges)
        row, col = np.transpose(edges)
        
        # duplicate edges in the opposite direction
        if include_inverse:
            _row = np.hstack([row, col])
            col = np.hstack([col, row])
            row = _row
            n_edges *= 2
        
        # create adjacency matrix for this predicate
        data = np.ones(n_edges)
        adj = sp.csr_matrix((data, (row, col)), shape=adj_shape)
        sp_adjacencies.append(adj)
    
    return np.asarray(sp_adjacencies)


from sklearn.preprocessing import normalize, binarize


def hop(entities, constraints, top_predicates, max_triples=200000, max_degree=100000):
    '''
    Extract the subgraph for the selected entities
    ''' 
#     print(top_predicates)
    n_constraints = len(constraints)
    if entities:
        n_constraints += 1

    top_entities = entities + constraints
#     all_entities_ids = [_id for e in top_entities for _id in e]
    top_predicates_ids = [_id for p in top_predicates for _id in p if _id]

    # skip heavy hitters
    all_entities_ids = []
    for e in top_entities:
        for _id in e:
            entity = e_index.look_up_by_id(_id)
            if entity:
                if int(entity[0]['_source']['count']) <= max_degree:
                    all_entities_ids.append(_id)
    if not all_entities_ids:
        return []

    # iteratively call the HDT API to retrieve all subgraph partitions
    activations = defaultdict(int)
    offset = 0

    while True:
        # get the subgraph for selected predicates only
        kg = HDTDocument(hdt_path+hdt_file)
#         print(top_predicates_ids)
        kg.configure_hops(1, top_predicates_ids, namespace, True)
        entities, predicate_ids, adjacencies = kg.compute_hops(all_entities_ids, max_triples, offset)
        kg.remove()

        if not entities:
            answers = [{a_id: a_score} for a_id, a_score in activations.items()]
            return answers

        offset += max_triples
        # index entity ids global -> local
        entities_dict = {k: v for v, k in enumerate(entities)}
        # generate a list of adjacency matrices per predicate assuming the graph is undirected wo self-loops
        A = generate_adj_sp(adjacencies, len(entities), include_inverse=True)
        # activate entities -- build sparse matrix
        row, col, data = [], [], []
        for i, concept_ids in enumerate(top_entities):
            for entity_id, score in concept_ids.items():
                if entity_id in entities_dict:
#                     print(e_index.look_up_by_id(entity_id)[0]['_source']['uri'])
#                     print(score)
                    local_id = entities_dict[entity_id]
                    row.append(i)
                    col.append(local_id)
                    data.append(score)
        x = sp.csr_matrix((data, (row, col)), shape=(len(top_entities), len(entities)))
    
        # iterate over predicates
        ye = sp.csr_matrix((len(top_entities), len(entities)))
        # activate predicates
        if top_predicates_ids:
            yp = sp.csr_matrix((len(top_predicates), len(entities)))
            for i, concept_ids in enumerate(top_predicates):
                # activate predicates
                p = np.zeros([len(predicate_ids)])
                # iterate over synonyms
                for p_id, score in concept_ids.items():
                    if p_id in predicate_ids:
                        local_id = predicate_ids.index(p_id)
                        p[local_id] = score
                # slice A by the selected predicates
                _A = sum(p*A)
                _y = x @ _A
                # normalize: cut top to 1
                _y[_y > 1] = 1
                yp[i] = _y.sum(0)
                ye += _y
            y = sp.vstack([ye,yp])
        # fall back to evaluate all predicates
        else:
            y = x @ sum(A)
        sum_a = sum(y)
        sum_a_norm = sum_a.toarray()[0] / (len(top_predicates) + n_constraints) #normalize(sum_a, norm='max', axis=1).toarray()[0]
        # normalize: cut top to 1
        sum_a_norm[sum_a_norm > 1] = 1
        # activations across components
        y_counts = binarize(y, threshold=0.0)
        count_a = sum(y_counts).toarray()[0]
        # final scores
        y = (sum_a_norm + count_a) / (len(top_predicates) + n_constraints + 1)

        # check output size
        assert y.shape[0] == len(entities)

        top = np.argwhere(y > 0).T.tolist()[0]
        if len(top) > 0:
            activations1 = np.asarray(entities)[top]
            # store the activation values per id answer id
            for i, e in enumerate(entities):
                if e in activations1:
                    activations[e] += y[i]
        # if not such answer found fall back to return the answers satisfying max of the constraints
        else:
            # select answers that satisfy maximum number of constraints
            y_p = np.argmax(y)
            # maximum number of satisfied constraints
            max_cs = y[y_p]
            # at least some activation (evidence from min one constraint)
            if max_cs != 0:
                # select answers
                top = np.argwhere(y == max_cs).T.tolist()[0]
                activations1 = np.asarray(entities)[top]
                # store the activation values per id answer id
                for i, e in enumerate(entities):
                    if e in activations1:
                        activations[e] += y[i]

# hold average stats for the model performance over the samples
from collections import Counter

limit = None

question_types = ['SELECT', 'ASK', 'COUNT']

# embeddings
word_vectors = load_embeddings(embeddings_choice)
p_vectors = load_embeddings('fasttext_p_labels')

# type predicates
bl_p = [68655]

cursor = mongo.get_sample(train=False, limit=limit)

with cursor:
    print("Evaluating...")
    for doc in cursor:
        q = doc['question']
        print(doc['SerialNumber'])

        # parse question into words and embed
        x_test_sent = np.zeros((model_settings['max_len'], model_settings['emb_dim']))
        q_words = text_to_word_sequence(q)
        for i, word in enumerate(q_words):
            x_test_sent[i] = word_vectors.query(word)
        
        # predict question type
        y_p = qt_model.predict(np.array([x_test_sent]))
        y_p = np.argmax(y_p, axis=-1)[0]
        p_qt = question_types[y_p]
        ask_question = p_qt == 'ASK'
        
        # use GS spans + preprocess
        y_p = ep_model.predict(np.array([x_test_sent]))
        y_p = np.argmax(y_p, axis=-1)[0]
        e_spans1 = collect_mentions(q_words, y_p, 1)
        p_spans1 = collect_mentions(q_words, y_p, 2)
        p_spans2 = collect_mentions(q_words, y_p, 3)

        # match predicates
        top_predicates_ids1 = relation_detection(p_spans1, threshold=0)
        top_predicates_ids2 = relation_detection(p_spans2, threshold=0)

        top_entities_ids1 = entity_linking(e_spans1, threshold=0.7)

        if ask_question:
            a_threshold = 0.0
        else:
            a_threshold = 0.5

        # MP
        answers_ids = []
            
        # 1st hop
        answers_ids1 = hop([], top_entities_ids1, top_predicates_ids1)
        answers1 = [{a_id: a_score} for activations in answers_ids1 for a_id, a_score in activations.items() if a_score > a_threshold]
        
        # 2nd hop
        if top_predicates_ids1 and top_predicates_ids2:
            answers_ids = hop(answers1, [], top_predicates_ids2)
            answers = [{a_id: a_score} for activations in answers_ids for a_id, a_score in activations.items() if a_score > a_threshold]
        else:
            answers = answers1

        answers_ids = [_id for a in answers for _id in a]
