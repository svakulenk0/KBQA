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
# namespace = "http://dbpedia.org/"
namespace = "predef-dbpedia2016-04"

import time
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
import pickle as pkl
with open('%s_%s.pkl'%(dataset_name, embeddings_choice), 'rb') as f:
    ep_model_settings = pkl.load(f)

ep_model = build_ep_inference_model(ep_model_settings)
# load weights
# ep_model.load_weights('checkpoints/_'+modelname+'_weights.best.hdf5', by_name=True)
ep_model.load_weights('model/'+modelname+'.h5', by_name=True)

# functions for entity linking and relation detection
def entity_linking(e_spans, verbose=False, cutoff=500, threshold=0): 
    guessed_ids = []
    for span in e_spans:
        span_ids = e_index.label_scores(span, top=cutoff, threshold=threshold, verbose=verbose, scale=0.3, max_degree=100000)
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
kg = HDTDocument(hdt_path+hdt_file)

def hop(entities, constraints, top_predicates, verbose=False, max_triples=500000):
    '''
    Extract the subgraph for the selected entities
    ''' 
#     print(top_predicates)
    n_constraints = len(constraints)
    if entities:
        n_constraints += 1

    top_entities = entities + constraints
    all_entities_ids = [_id for e in top_entities for _id in e]
    top_predicates_ids = [_id for p in top_predicates for _id in p if _id]
            

    # iteratively call the HDT API to retrieve all subgraph partitions
    activations = defaultdict(int)
    offset = 0

    while True:
        # get the subgraph for selected predicates only
#         print(top_predicates_ids)
        kg.configure_hops(1, top_predicates_ids, namespace, True)
        entities, predicate_ids, adjacencies = kg.compute_hops(all_entities_ids, max_triples, offset)
#         print(adjacencies)
        # show subgraph entities
#         print([e_index.look_up_by_id(e)[0]['_source']['uri'] for e in entities])
        
        if not entities:
            answers = [{a_id: a_score} for a_id, a_score in activations.items()]
            return answers

        if verbose:
            print("Subgraph extracted:")
            print("%d entities"%len(entities))
            print("%d predicates"%len(predicate_ids))
            print("Loading adjacencies..")

        offset += max_triples
        # index entity ids global -> local
        entities_dict = {k: v for v, k in enumerate(entities)}
        # generate a list of adjacency matrices per predicate assuming the graph is undirected wo self-loops
        A = generate_adj_sp(adjacencies, len(entities), include_inverse=True)
#         print(predicate_ids)
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

verbose = False
limit = None

question_types = ['SELECT', 'ASK', 'COUNT']

# embeddings
word_vectors = load_embeddings(embeddings_choice)
p_vectors = load_embeddings('fasttext_p_labels')

errors_1 = ['67', '138', '392', '467', '563', '581', '601', '723', '741', '785', '920', '951', '952', '1029', '1070', '1140', '1142', '1149', '1219', '1253', '1325', '1461', '1485', '1620', '1626', '1640', '1669', '1680', '1687', '1762', '1866', '1918', '2039', '2191', '2205', '2395', '2398', '2455', '2547', '2557', '2569', '2613', '2732', '2739', '2745', '2833', '2854', '2872', '2873', '2983', '3142', '3267', '3282', '3288', '3295', '3450', '3458', '3466', '3487', '3508', '3738', '3757', '3767', '3792', '3799', '3813', '3824', '3864', '3944', '3975', '4034', '4068', '4125', '4139', '4172', '4219', '4339', '4352', '4418', '4465', '4466', '4486', '4487', '4489', '4566', '4683', '4703', '4724', '4729', '4732', '4739']
errors_e = ['25', '56', '118', '126', '128', '134', '147', '162', '468', '475', '489', '538', '609', '624', '636', '646', '647', '816', '873', '976', '981', '1063', '1078', '1218', '1276', '1292', '1343', '1446', '1529', '1636', '1637', '1683', '1697', '1807', '1835', '1839', '1932', '1997', '1998', '2028', '2057', '2076', '2106', '2266', '2364', '2377', '2394', '2450', '2473', '2503', '2505', '2524', '2564', '2566', '2694', '2704', '2731', '2740', '2784', '2807', '2870', '2892', '2912', '2929', '2959', '3000', '3058', '3070', '3100', '3118', '3172', '3206', '3213', '3237', '3302', '3374', '3386', '3395', '3446', '3462', '3473', '3515', '3518', '3562', '3578', '3638', '3646', '3697', '3953', '3954', '3988', '3994', '4062', '4143', '4262', '4286', '4345', '4346', '4390', '4399', '4433', '4518', '4536', '4599', '4601', '4660', '4707', '4821', '4844', '4854', '4880', '4961', '4972']
# class_constraints = True

# type predicates
bl_p = [68655]

ps, rs, ts = [], [], []
nerrors = 0
errors_ids = []
n_missing_entities = 0
qt_errors = 0
n_missing_spans = 0

new_answers = ['134', '1839', '2450', '3213', '3237', '3302', '4390', '4972']

cursor = mongo.get_sample(train=False, limit=limit)
# cursor = mongo.get_by_id('63', limit=1)
with cursor:
    print("Evaluating...")

    # start = time.time()
    for doc in cursor:
        print(doc['SerialNumber'])
#         if doc_id not in new_answers:
#             continue
        
        start_one = time.time()
        q = doc['question']
                
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

#         c_spans1 = doc['c1_spans']
#         c_spans2 = doc['c2_spans']
        
        # match predicates
        top_predicates_ids1 = relation_detection(p_spans1, threshold=0)
        top_predicates_ids2 = relation_detection(p_spans2, threshold=0)

        # use GS classes
#         classes1 = [{_id: 1} for _id in doc['classes_ids'] if _id in doc['1hop_ids'][0]]
#         classes2 = [{_id: 1} for _id in doc['classes_ids'] if _id in doc['2hop_ids'][0]]
        
        top_entities_ids1 = entity_linking(e_spans1, threshold=0.7)

        if ask_question:
            a_threshold = 0.0
        else:
            a_threshold = 0.5

        # MP
        answers_ids = []
            
        # 1st hop
        answers_ids1 = hop([], top_entities_ids1, top_predicates_ids1, verbose)
#         if classes1:
#             answers_ids1 = filter_answer_by_class(classes1, answers_ids1)
        answers1 = [{a_id: a_score} for activations in answers_ids1 for a_id, a_score in activations.items() if a_score > a_threshold]
        
        # 2nd hop
        if top_predicates_ids1 and top_predicates_ids2:                
            answers_ids = hop(answers1, [], top_predicates_ids2, verbose)
#             if classes2:
#                 answers_ids = filter_answer_by_class(classes2, answers_ids)
            answers = [{a_id: a_score} for activations in answers_ids for a_id, a_score in activations.items() if a_score > a_threshold]
        else:
            answers = answers1

        answers_ids = [_id for a in answers for _id in a]

        # error estimation
        if p_qt != doc['question_type']:
            nerrors += 1
            # print(doc['SerialNumber'], doc['question'])
            # print("%s instead of %s"%(p_qt, doc['question_type']))
            # incorrect question type
            qt_errors += 1
            p, r = 0, 0
        else:
            all_entities_baskets = [set(e.keys()) for e in top_entities_ids1]
            # ASK
            if ask_question:
                # make sure the output matches every input basket
                answer = all(x & set(answers_ids) for x in all_entities_baskets)
                # compare with GS answer
                gs_answer = doc['bool_answer']
                gs_answer_ids = []
                if answer == gs_answer:
                    p, r = 1, 1
                else:
                    p, r = 0, 0
            else:
                answers_ids = set(answers_ids)
                n_answers = len(answers_ids)
                # compare with GS answer
                gs_answer_ids = set(doc['answers_ids'])
                n_gs_answers = len(gs_answer_ids)
                
                # COUNT
                if p_qt == 'COUNT':
                    if n_answers == n_gs_answers:
                        p, r = 1, 1
                    else:
                        p, r = 0, 0

                # SELECT
                else:
                    n_correct = len(answers_ids & gs_answer_ids)
                    try:
                        r = float(n_correct) / n_gs_answers
                    except ZeroDivisionError:\
                        print(doc['question'])
                    try:
                        p = float(n_correct) / n_answers
                    except ZeroDivisionError:
                        p = 0
            
            if p < 1.0 or r < 1.0: 
#             if True:
                nerrors += 1
#                 if doc['SerialNumber'] not in errors_1+errors_e:
#                     if doc['SerialNumber'] not in errors_1:
#                         errors_ids.append(doc['SerialNumber'])

#                         # GS entities
#                         gs_classes_ids1 = [_id for _id in doc['1hop_ids'][1] if _id not in bl_p]
#                         gs_classes_ids2 = [_id for _id in doc['2hop_ids'][1] if _id not in bl_p]
                        
#                         gs_e_ids2 = [_id for _id in doc['1hop_ids'][0] if _id not in bl_p]


#                         # check the number of detected concepts is correct
#             #             assert len(gs_top_entities_ids1) == len(all_entities_baskets)
#                         all_entities_ids = [_id for e in top_entities_ids1 for _id in e]
#                         if (gs_classes_ids1 or gs_classes_ids2) and not all_entities_ids:
#                             n_missing_spans += 1
#                         missed = False
#                         for e in gs_e_ids2:
#                             if e not in all_entities_ids:
#                                 missed = True
#                                 break

#                         # find missing entity matches
#                         all_entities_ids = [_id for e in top_predicates_ids1+top_predicates_ids2 for _id in e]
#                         if (gs_classes_ids1 or gs_classes_ids2) and not all_entities_ids:
#                             n_missing_spans += 1
#                         missed = False
#                         for e in gs_classes_ids1+gs_classes_ids2:
#                             if e not in all_entities_ids:
#                                 missed = True
#                                 break
#                                 e = p_index.look_up_by_id(e)
#                                 if e:
#                                     print(doc['SerialNumber'], doc['question'])
#                                     print(doc['sparql_query'])
#                                     print("Missing predicate match: %s"%e[0]['_source']['uri'])
#                         if missed:
#                             n_missing_entities += 1
# #                         else:
#                         print(doc['SerialNumber'], doc['question'])
#                         print(doc['sparql_query'])
                        # show spans
#                             print(p_spans1)
#                             print(p_spans2)

                        # show  matches
#                             print([{p_index.look_up_by_id(_id)[0]['_source']['uri']: score} for answer in top_predicates_ids1+top_predicates_ids2 for _id, score in answer.items() if p_index.look_up_by_id(_id) ])

                        # show answers before applying activation threshold
#                             print(answers_ids1)

#                             # show intermediate answers if there was a second hop
#                             if top_predicates_ids2:
#                                 print([{e_index.look_up_by_id(_id)[0]['_source']['uri']: score} for answer in answers1 for _id, score in answer.items() if e_index.look_up_by_id(_id)])

#                             # show correct answers
#                             print([{e_index.look_up_by_id(_id)[0]['_source']['uri']: score} for answer in answers for _id, score in answer.items() if _id in gs_answer_ids if e_index.look_up_by_id(_id)])

#                             # show errors            
#                             print([{e_index.look_up_by_id(_id)[0]['_source']['uri']: score} for answer in answers for _id, score in answer.items() if _id not in gs_answer_ids if e_index.look_up_by_id(_id)])
                        # print('\n')

        # add stats
        ps.append(p)
        rs.append(r)
        ts.append(time.time() - start_one)

# show basic stats
min_len = min(ts)
mean_len = np.mean(ts)
median_len = np.median(ts)
max_len = max(ts)
print("Min:%.2f Median:%.2f Mean:%.2f Max:%.2f"%(min_len, median_len, mean_len, max_len))

# print("--- %.2f seconds ---" % (float(time.time() - start)/999))
print("\nFin. Results for %d questions:"%len(ps))
print("P: %.2f R: %.2f"%(np.mean(ps), np.mean(rs)))
print("Number of errors: %d"%nerrors)
print(errors_ids)
print("Number of questions with missing entity matches: %d"%n_missing_entities)
print("Number of questions with incorrect question type detection: %d"%qt_errors)
print("Number of questions with missing spans: %d"%n_missing_spans)
