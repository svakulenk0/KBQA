# setup
dataset_name = 'lcquad'
embeddings_choice = 'glove840B300d'

question_types = ['SELECT', 'ASK', 'COUNT']

# connect to DB storing the dataset
from setup import Mongo_Connector, load_embeddings, IndexSearch
mongo = Mongo_Connector('kbqa', dataset_name)

# entity and predicate catalogs
e_index = IndexSearch('dbpedia201604e')
p_index = IndexSearch('dbpedia201604p')

# path to KG relations
from hdt import HDTDocument
hdt_path = "/home/zola/Projects/hdt-cpp-molecules/libhdt/data/"
hdt_file = 'dbpedia2016-04en.hdt'
namespace = "http://dbpedia.org/"
    
word_vectors = load_embeddings(embeddings_choice)

from collections import defaultdict

import numpy as np
import scipy.sparse as sp

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
modelname = '2hops'

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
    model.compile(optimizer=Nadam(clipnorm=1), loss=losses.crf_loss, metrics=[metrics.crf_accuracy]) 
    model.summary()
    return model

# load model settings
import pickle as pkl
with open('%s_%s.pkl'%(dataset_name, embeddings_choice), 'rb') as f:
    ep_model_settings = pkl.load(f)

ep_model = build_ep_inference_model(ep_model_settings)
# load weights
ep_model.load_weights('checkpoints/_'+modelname+'_weights.best.hdf5', by_name=True)

# load MP functions

def generate_adj_sp(adjacencies, adj_shape, normalize=False, include_inverse=False):
    '''
    Build adjacency matrix
    '''
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
        
        # create adjacency matrix for this predicate TODO initialise matrix with predicate scores
        data = np.ones(n_edges, dtype=np.int8)
        adj = sp.csr_matrix((data, (row, col)), shape=adj_shape, dtype=np.int8)
        
        if normalize:
            adj = normalize_adjacency_matrix(adj)

        sp_adjacencies.append(adj)
    
    return np.asarray(sp_adjacencies)


def hop(activations, constraints, predicates_ids, verbose=False, _bool_answer=False, max_triples=500000):
    # extract the subgraph for the selected entities
    top_entities_ids = [_id for e in activations + constraints for _id in e]
    # exclude types predicate
    top_predicates_ids = [_id for p in predicates_ids for _id in p if _id != 68655]


    # iteratively call the HDT API to retrieve all subgraph partitions
    activations = defaultdict(int)
    offset = 0
    while True:
        # get the subgraph for selected predicates only
        kg = HDTDocument(hdt_path+hdt_file)
        kg.configure_hops(1, top_predicates_ids, namespace, True)
        entities, predicate_ids, adjacencies = kg.compute_hops(top_entities_ids, max_triples, offset)
        kg.remove()
    
        if not entities:
            # filter out the answers by min activation scores
            if not _bool_answer and constraints:
                # normalize activations by checking the 'must' constraints: number of constraints * weights
                min_a = len(constraints) * 1
                if predicates_ids != top_predicates_ids:
                    min_a -= 1
            else:
                min_a = 0
            # return HDT ids of the activated entities
            return [a_id for a_id, a_score in activations.items() if a_score > min_a]

        if verbose:
            print("Subgraph extracted:")
            print("%d entities"%len(entities))
            print("%d predicates"%len(predicate_ids))
            print("Loading adjacencies..")

        offset += max_triples
        # index entity ids global -> local
        entities_dict = {k: v for v, k in enumerate(entities)}
        adj_shape = (len(entities), len(entities))
        # generate a list of adjacency matrices per predicate assuming the graph is undirected wo self-loops
        A = generate_adj_sp(adjacencies, adj_shape, include_inverse=True)

        # activations of entities and predicates
        e_ids = [entities_dict[entity_id] for entity_id in top_entities_ids if entity_id in entities_dict]
    #     assert len(top_entities_ids) == len(e_ids)
        p_ids = [predicate_ids.index(entity_id) for entity_id in top_predicates_ids if entity_id in predicate_ids]
    #     assert len(top_predicates_ids) == len(p_ids)
        if p_ids:
            # graph activation vectors
            x = np.zeros(len(entities))
            x[e_ids] = 1
            p = np.zeros(len(predicate_ids))
            p[p_ids] = 1

            # slice A by the selected predicates and concatenate edge lists
            y = (x@sp.hstack(A*p)).reshape([len(predicate_ids), len(entities)]).sum(0)
            # check output size
            assert y.shape[0] == len(entities)
            
            # harvest activations
            top = np.argwhere(y > 0).T.tolist()[0]
            if len(top) > 0:
                activations1 = np.asarray(entities)[top]
                # store the activation values per id answer id
                for i, e in enumerate(entities):
                    if e in activations1:
                        activations[e] += y[i]

print("MP functions loaded.")

# functions for entity linking and relation detection
def entity_linking(e_spans, verbose=False, string_cutoff=50): 
    guessed_ids = []
    for span in e_spans:
        span_ids = []
        for match in e_index.match_label(span, top=string_cutoff):
            _id = match['_source']['id']
            span_ids.append(_id)
            if verbose:
                uri = match['_source']['uri']
                print(uri)
        guessed_ids.append(span_ids)
    return guessed_ids


def relation_detection(p_spans, verbose=False, string_cutoff=50): 
    guessed_ids = []
    for span in p_spans:
        span_ids = []
        for match in p_index.match_label(span, top=string_cutoff):
            _id = match['_source']['id']
            span_ids.append(_id)
            if verbose:
                uri = match['_source']['uri']
                print(uri)
        guessed_ids.append(span_ids)
    return guessed_ids


print("Linking functions loaded.")

# get test data split
verbose = False
limit = None

# vector of choices for using Gold Standard annotations:
# question type, entity ids, predicate ids
gs = [False, False, False]

# hold average stats for the model performance over the samples
ps, rs, fs = [], [], []

cursor = mongo.get_sample(train=True, limit=limit)
with cursor:
    print("Evaluating...")
    for doc in cursor:
        q = doc['question']
        if verbose:
            print(q)
        # parse question into words and embed
        x_test_sent = np.zeros((model_settings['max_len'], model_settings['emb_dim']))
        q_words = text_to_word_sequence(q)
        for i, word in enumerate(q_words):
            x_test_sent[i] = word_vectors.query(word)
        
        # question type
        if gs[0]:
            _bool_answer = doc['question_type'] == 'ASK'
            _count_answer = doc['question_type'] == 'COUNT'
        else:
            # guess question type
            y_p = qt_model.predict(np.array([x_test_sent]))
            y_p = np.argmax(y_p, axis=-1)[0]
            p = question_types[y_p]
#             print(p)
            _bool_answer = p == 'ASK'
            _count_answer = p == 'COUNT'
        
        # parse the question into spans
        if not gs[1] or not gs[2]:
            y_p = ep_model.predict(np.array([x_test_sent]))
            y_p = np.argmax(y_p, axis=-1)[0]
            e_spans1 = collect_mentions(q_words, y_p, 1)
            p_spans1 = collect_mentions(q_words, y_p, 2)
            e_spans2 = collect_mentions(q_words, y_p, 3)
            p_spans2 = collect_mentions(q_words, y_p, 4)

        # entities
        if gs[1]:
            top_entities_ids1 = [[_id] for _id in doc['1hop_ids'][0]]
        else:
            # guess entities
            if verbose:
                print(e_spans1)
                print(doc['1hop'])
            top_entities_ids1 = entity_linking(e_spans1, verbose)
        
        # predicates
        if gs[2]:
            top_predicates_ids1 = [[_id] for _id in doc['1hop_ids'][1]]
        else:
            # guess predicates
            top_predicates_ids1 = relation_detection(p_spans1, verbose)
        
        # MP
        answers_ids = []
        if top_entities_ids1 and top_predicates_ids1:
            answers_ids = hop([top_entities_ids1[0]], top_entities_ids1[1:], top_predicates_ids1, verbose, _bool_answer)
            
            _2hops = doc['2hop'] != [[], []]
            if _2hops:
                # entities
                if gs[1]:
                    top_entities_ids2 = [[_id] for _id in doc['2hop_ids'][0]]
                else:
                    # guess entities
                    top_entities_ids2 = entity_linking(e_spans2, verbose)


                # predicates
                if gs[2]:
                    top_predicates_ids2 = [[_id] for _id in doc['2hop_ids'][1]]
                else:
                    # guess predicates
                    top_predicates_ids2 = relation_detection(p_spans2, verbose)

                # MP
                if answers_ids and top_predicates_ids2:
                    answers_ids = [[_id] for _id in answers_ids]
                    answers_ids = hop(answers_ids, top_entities_ids2, top_predicates_ids2, verbose)


        # error estimation
        
        # ASK
        if _bool_answer:
            answer = all(x in answers_ids for x in doc["entity_ids"])
            if 'bool_answer' in doc:
                gs_answer = doc['bool_answer']
                if answer == gs_answer:
                    p, r, f = 1, 1, 1
                else:
                    p, r, f = 0, 0, 0
            # mistake identifying the question type
            else:
                p, r, f = 0, 0, 0
        
        else:
            answers_ids = set(answers_ids)
            n_answers = len(answers_ids)
            if 'answers_ids' in doc:
                gs_answer_ids = set(doc['answers_ids'])
                n_gs_answers = len(gs_answer_ids)

                # COUNT
                if _count_answer:
                    if n_answers == n_gs_answers:
                        p, r, f = 1, 1, 1
                    else:
                        p, r, f = 0, 0, 0

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
                    try:
                        f = 2 * p * r / (p + r)
                    except ZeroDivisionError:
                        f = 0
            # mistake identifying the question type
            else:
                p, r, f = 0, 0, 0
        print("P: %.2f R: %.2f F: %.2f"%(p, r, f))

        # add stats
        ps.append(p)
        rs.append(r)
        fs.append(f)


print("\nFin. Results for %d questions:"%len(ps))
print("P: %.2f R: %.2f F: %.2f"%(np.mean(ps), np.mean(rs), np.mean(fs)))