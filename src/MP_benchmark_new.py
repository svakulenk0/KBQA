#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Dec 9, 2018

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Message Passing for KBQA
'''

# setup
dataset_name = 'lcquad'
kg_name = 'dbpedia201604'

# connect to MongoDB (27017 is the default port) to access the dataset
# sudo service mongod start 
from pymongo import MongoClient


class Mongo_Connector():
    '''
    Wrapper class for some of the pymongo functions: http://api.mongodb.com/python/current/tutorial.html
    '''

    def __init__(self, db_name, col_name):
        # spin up database
        self.mongo_client = MongoClient()
        self.db = self.mongo_client[db_name]
        self.col = self.db[col_name]
        
    def get_sample(self, limit=100):
        '''
        Set limit to None to get all docs
        '''
        cursor = self.col.find({'question_type': {'$ne': 'ASK'}, 'train': True}, no_cursor_timeout=True)
        if limit:
            cursor = cursor.limit(limit)
        return cursor
    
    def get_by_id(self, id, limit=1):
        '''
        '''
        cursor = self.col.find({'SerialNumber': id})
        if limit:
            cursor = cursor.limit(limit)
        return cursor[0]


mongo = Mongo_Connector('kbqa', dataset_name)

# path to KG relations
from hdt import HDTDocument
hdt_path = "/home/zola/Projects/hdt-cpp-molecules/libhdt/data/"
hdt_file = 'dbpedia2016-04en.hdt'
namespace = "http://dbpedia.org/"

# connect to entity catalog indexed with Lucene 
from elasticsearch import Elasticsearch
from urllib.parse import quote
import string

class IndexSearch:
    
    def __init__(self, index_name):
        # set up ES connection
        self.es = Elasticsearch()
        self.index = index_name
        self.type = 'terms'

    def match_label(self, string, top=100):
        return self.es.search(index=self.index,
                              body={"query": {"multi_match": {"query": string,
                                                              "operator": "and",
                                                              "fields": ["label^10", "label.ngrams"],
                                                              }}},
                              size=top, doc_type=self.type)['hits']['hits']

    def look_up_by_id(self, _id, top=1):
        results = self.es.search(index=self.index,
                              body={"query": {"term": {"id": _id}}},
                              size=top, doc_type=self.type)['hits']['hits']
        return results
    
    def look_up_by_uri(self, uri, top=10):
#         results = self.es.search(index=self.index,
#                               body={"query": {"term": {"uri": quote(uri, safe=string.punctuation)}}},
#                               size=top, doc_type=self.type)['hits']['hits']
#         if not results:
        results = self.es.search(index=self.index,
                          body={"query": {"term": {"uri": uri}}},
                          size=top, doc_type=self.type)['hits']['hits']

        return results


e_index = IndexSearch('%se'%kg_name)
p_index = IndexSearch('%sp'%kg_name)


# hop hop

import numpy as np
import scipy.sparse as sp


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


def hop(activations, constraints, top_predicates_ids, verbose=False):
    # extract the subgraph
    top_entities_ids = activations + constraints
    kg = HDTDocument(hdt_path+hdt_file)
    kg.configure_hops(1, [], namespace, True)
    entities, predicate_ids, adjacencies = kg.compute_hops(top_entities_ids)
    kg.remove()
    
    if verbose:
        print("Subgraph extracted:")
        print("%d entities"%len(entities))
        print("%d predicates"%len(predicate_ids))
    
    # index entity ids global -> local
    entities_dict = {k: v for v, k in enumerate(entities)}
    adj_shape = (len(entities), len(entities))
    # generate a list of adjacency matrices per predicate assuming the graph is undirected wo self-loops
    A = generate_adj_sp(adjacencies, adj_shape, include_inverse=True)
    
    # activations of entities and predicates
    e_ids = [entities_dict[entity_id] for entity_id in top_entities_ids if entity_id in entities_dict]
    assert len(top_entities_ids) == len(e_ids)
    p_ids = [predicate_ids.index(entity_id) for entity_id in top_predicates_ids if entity_id in predicate_ids]
    # assert len(top_predicates_ids) == len(p_ids)
    # missing entities due to incorrect unicode match
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
        
        if constraints:
            # normalize activations by checking the 'must' constraints: number of constraints * weights
            y -= len(constraints) * 1

        top = np.argwhere(y > 0).T.tolist()[0]
        
        # check activated entities
        if len(top) > 0:
            activations1 = np.asarray(entities)[top]
            activations1 = [int(entity_id) for entity_id in activations1.tolist()]
            # look up activated entities by ids
            activations1_labels = []
            for entity_id in activations1:
                matches = e_index.look_up_by_id(entity_id)
                for match in matches:
                    activations1_labels.append(match['_source']['uri'])
            
            # show predicted answer
            if verbose:
                print("%d answers"%len(top))
                print(activations1_labels[:5])
            
                # activation values
                scores = y[top]
                print(scores[:5])
            
            # return HDT ids of the activated entities
            return list(set(activations1)), list(set(activations1_labels))
    return [], []

limit = None
cursor = mongo.get_sample(limit=limit)
verbose = False

# hold average stats for the model performance over the samples
ps, rs, fs = [], [], []

with cursor:
    for doc in cursor:
        print(doc['SerialNumber'])

        if verbose:
            print(doc['question'])
            print(doc['sparql_query'])
            print(doc["1hop"])
            print(doc["2hop"])

        assert doc['train'] == True

        top_entities_ids1 = doc['1hop_ids'][0]
        top_predicates_ids1 = doc['1hop_ids'][1]
        answers_ids, answers_uris = hop([top_entities_ids1[0]], top_entities_ids1[1:], top_predicates_ids1, verbose=verbose)

        _2hops = doc['2hop'] != [[], []]
        if _2hops:
            top_entities_ids2 = doc['2hop_ids'][0]
            top_predicates_ids2 = doc['2hop_ids'][1]
            answers_ids, answers_uris = hop(answers_ids, top_entities_ids2, top_predicates_ids2, verbose=verbose)

        # error estimation
        answers_ids = set(answers_ids)
        n_answers = len(answers_ids)
        gs_answer_uris = set(doc['answers_ids'])
        n_gs_answers = len(gs_answer_uris)
        n_correct = len(answers_ids & gs_answer_uris)

        if verbose:
            print("%d predicted answers:"%n_answers)
            print(set(answers_uris))
            print("%d gs answers:"%n_gs_answers)
            print(set(doc['answers']))
            print(n_correct)

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
        print("P: %.2f R: %.2f F: %.2f"%(p, r, f))

        # add stats
        ps.append(p)
        rs.append(r)
        fs.append(f)


print("\nFin. Results for %d questions:"%len(ps))
print("P: %.2f R: %.2f F: %.2f"%(np.mean(ps), np.mean(rs), np.mean(fs)))
