#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Feb 9, 2018

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Evaluate MP algorithm with LCQUAD annotations
'''

# setup
dataset_name = 'lcquad'

# connect to DB storing the dataset
from setup import Mongo_Connector, load_embeddings, IndexSearch
mongo = Mongo_Connector('kbqa', dataset_name)

# path to KG relations
from hdt import HDTDocument
hdt_path = "/home/zola/Projects/hdt-cpp-molecules/libhdt/data/"
hdt_file = 'dbpedia2016-04en.hdt'
namespace = "http://dbpedia.org/"

from collections import defaultdict
import numpy as np
import scipy.sparse as sp

# entity and predicate catalogs
e_index = IndexSearch('dbpedia201604e')
p_index = IndexSearch('dbpedia201604p')

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
        data = np.ones(n_edges, dtype=np.int8)
        adj = sp.csr_matrix((data, (row, col)), shape=adj_shape, dtype=np.int8)
       
        sp_adjacencies.append(adj)
    
    return np.asarray(sp_adjacencies)


from sklearn.preprocessing import normalize, binarize


def hop(entities, constraints, top_predicates, verbose=False, max_triples=200000):
    '''
    Extract the subgraph for the selected entities
    ''' 
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
        kg = HDTDocument(hdt_path+hdt_file)
        kg.configure_hops(1, top_predicates_ids, namespace, True)
        entities, predicate_ids, adjacencies = kg.compute_hops(all_entities_ids, max_triples, offset)
        kg.remove()

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
        
        # activate entities -- build sparse matrix
        row, col, data = [], [], []
        for i, concept_ids in enumerate(top_entities):
            for entity_id, score in concept_ids.items():
                if entity_id in entities_dict:
                    local_id = entities_dict[entity_id]
                    row.append(i)
                    col.append(local_id)
                    data.append(score)
        x = sp.csr_matrix((data, (row, col)), shape=(len(top_entities), len(entities)), dtype=np.int8)
    
        # iterate over predicates
        ye = sp.csr_matrix((len(top_entities), len(entities)))
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
            yp[i] = _y.sum(0)
            ye += _y
        y = sp.vstack([ye,yp])
        
        sum_a = sum(y)
        sum_a_norm = normalize(sum_a, norm='max', axis=1).toarray()[0]

        # activations across components
        y_counts = binarize(y, threshold=0.0)
        count_a = sum(y_counts).toarray()[0]
        # final scores
#         y = (sum_a_norm + count_a) / (len(top_predicates) + n_constraints)
        y = count_a / (len(top_predicates) + n_constraints)
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


def filter_answer_by_class(classes, answers_ids):
    classes_ids = [_id for e in classes for _id in e]
    kg = HDTDocument(hdt_path+hdt_file)
    a_ids = [_id for e in answers_ids for _id in e]
    a_ids = kg.filter_types(a_ids, classes_ids)
    kg.remove()
    a_ids = [_id for _a_ids in a_ids for _id in _a_ids]
    answers_ids = [{_id: a_score} for e in answers_ids for _id, a_score in e.items() if _id in a_ids]
    return answers_ids
        

print("MP functions loaded.")

# hold average stats for the model performance over the samples
verbose = False
limit = None

# type predicates
bl_p = [68655]

ps, rs = [], []
nerrors = 0
cursor = mongo.get_sample(train=False, limit=limit)

cursor = mongo.get_by_id('1762', limit=1)
with cursor:
    print("Evaluating...")
    for doc in cursor:
#         print(doc['SerialNumber'])
        q = doc['question']
        if verbose:
            print(q)
        
        # use GS annotation for entities classes and predicates across hops
        top_entities_ids1 = [{_id: 1} for _id in doc['1hop_ids'][0] if _id not in doc['classes_ids']]
        classes1 = [{_id: 1} for _id in doc['classes_ids'] if _id in doc['1hop_ids'][0]]
        classes2 = [{_id: 1} for _id in doc['classes_ids'] if _id in doc['2hop_ids'][0]]
        top_predicates_ids1 = [{_id: 1} for _id in doc['1hop_ids'][1] if _id not in bl_p]
        top_predicates_ids2 = [{_id: 1} for _id in doc['2hop_ids'][1] if _id not in bl_p]
        
        
        if doc['question_type'] == 'ASK':
            a_threshold = 0.0
        else:
            a_threshold = 0.9
            

        # MP
        answers_ids = []
            
        # 1st hop
        answers_ids1 = hop([], top_entities_ids1, top_predicates_ids1, verbose)
        if classes1:
            answers_ids1 = filter_answer_by_class(classes1, answers_ids1)
        answers1 = [{a_id: a_score} for activations in answers_ids1 for a_id, a_score in activations.items() if a_score > a_threshold]
        
        # 2nd hop
        if top_predicates_ids2:                
            answers_ids = hop(answers1, [], top_predicates_ids2, verbose)
            if classes2:
                answers_ids = filter_answer_by_class(classes2, answers_ids)
            answers = [{a_id: a_score} for activations in answers_ids for a_id, a_score in activations.items() if a_score > a_threshold]
        else:
            answers = answers1

        answers_ids = [_id for a in answers for _id in a]


#         answers = [{a_id: a_score} for activations in answers_ids for a_id, a_score in activations.items() if a_score > a_threshold]
#         answers_ids = [_id for a in answers for _id in a]
        
        # error estimation
        # ASK
        if doc['question_type'] == 'ASK':
            answer = all(x in answers_ids for x in doc["entity_ids"])
            gs_answer = doc['bool_answer']
            if answer == gs_answer:
                p, r, f = 1, 1, 1
            else:
                p, r, f = 0, 0, 0
        else:
            answers_ids = set(answers_ids)
            n_answers = len(answers_ids)
            gs_answer_ids = set(doc['answers_ids'])
            n_gs_answers = len(gs_answer_ids)
            if verbose:
                # show the scores for correct answers
                print([{e_index.look_up_by_id(_id)[0]['_source']['uri']: score} for answer in answers for _id, score in answer.items() if _id in gs_answer_ids])
                # show only new answers
                print([{e_index.look_up_by_id(_id)[0]['_source']['uri']: score} for answer in answers for _id, score in answer.items() if _id not in gs_answer_ids])

            # SELECT (COUNT as well)
            n_correct = len(answers_ids & gs_answer_ids)
            try:
                r = float(n_correct) / n_gs_answers
            except ZeroDivisionError:\
                print(doc['question'])
            try:
                p = float(n_correct) / n_answers
            except ZeroDivisionError:
                p = 0

#         print("P: %.2f R: %.2f\n"%(p, r))
        # add stats
        ps.append(p)
        rs.append(r)
        
        if p < 1.0 or r < 1.0:
            nerrors += 1
            print(doc['SerialNumber'], doc['question'])
            print(doc['sparql_query'])
#             print(n_answers)
            
            # show intermediate answers
#             if top_predicates_ids2:
#                 print([{e_index.look_up_by_id(_id)[0]['_source']['uri']: score} for answer in answers1 for _id, score in answer.items() if _id not in gs_answer_ids])
            
            # show correct answers
#             print([{e_index.look_up_by_id(_id)[0]['_source']['uri']: score} for answer in answers for _id, score in answer.items() if _id in gs_answer_ids])
#             print(doc['answers'])
            # show errors
            print([{e_index.look_up_by_id(_id)[0]['_source']['uri']: score} for answer in answers for _id, score in answer.items() if _id not in gs_answer_ids if e_index.look_up_by_id(_id)])
#             print('\n')

print("\nFin. Results for %d questions:"%len(ps))
print("P: %.2f R: %.2f"%(np.mean(ps), np.mean(rs)))
print("Number of errors: %d"%nerrors)