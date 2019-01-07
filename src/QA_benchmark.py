#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Dec 29, 2018

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

1-hop MP for KBQA using annotations from MongoDB
'''

# setup
dataset_name = 'lcquad'
kg_name = 'dbpedia201604'

limit = None
nhops = 1

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
        cursor = self.col.find()
        if limit:
            cursor = cursor.limit(limit)
        return cursor


mongo = Mongo_Connector('kbqa', dataset_name)

# path to KG relations
from hdt import HDTDocument
hdt_path = "/home/zola/Projects/hdt-cpp-molecules/libhdt/data/"
hdt_file = 'dbpedia2016-04en.hdt'
namespace = "http://dbpedia.org/"

# connect to entity catalog
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

    def look_up_by_uri(self, uri, top=1):
        results = self.es.search(index=self.index,
                                 body={"query": {"term": {"uri": quote(uri, safe=string.punctuation)}}},
                                 size=top, doc_type=self.type)['hits']['hits']

    def look_up_by_id(self, _id, top=1):
        results = self.es.search(index=self.index,
                                 body={"query": {"term": {"id": _id}}},
                                 size=top, doc_type=self.type)['hits']['hits']
        return results

    def match_entities(self, query=None, match_by="label", filter='terms', top=100):
        '''
        Index search
        size – Number of hits to return (default: 10)
        '''
        if query:
            if match_by == "label":
                results = self.es.search(index=self.index,
                                         body={"query": {"match": {match_by: {"query": query, "fuzziness": "AUTO"}}}},
                                         size=top,
                                         # body={"query": {"match": {match_by: {"query": query, "operator" : "and", "fuzziness": "AUTO"}}}},
                                         doc_type=self.type)['hits']
            
            elif match_by == "uri" or match_by == "id":
                # filter out only entities in s and o positions
                results = self.es.search(index=self.index,
                                         body={
                                              "query": {
                                                "constant_score": {
                                                  "filter": {
                                                      "term": {
                                                        match_by: query
                                                      }
                                                    }
                                                  }
                                                }
                                              },
                                         doc_type=self.type)['hits']
        else:
            # sample of size 2
            results = self.es.search(index=self.index, size=2, body={"query": {"match_all": {}}})['hits']
        return results['hits']


e_index = IndexSearch('%se'%kg_name)
p_index = IndexSearch('%sp'%kg_name)

# parse the subgraph into a sparse matrix
import numpy as np
import scipy.sparse as sp

def generate_adj_sp(adjacencies, adj_shape, normalize=False, include_inverse=False):
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

# get a context subgraph by seed entities and predicates sets

e_field, p_field = 'entity_guess', 'predicate_guess'

samples = mongo.get_sample(limit=limit)

# subgraph extraction accuracy
accs = []
# hold macro-average stats for the model performance over the samples
ps, rs, fs = [], [], []

# iterate over the cursor
for doc in samples:
    top_entities = doc[e_field]
    top_properties = doc[p_field]
    top_entities_ids = list(set([e_candidate['id'] for span in top_entities.values() for e_candidate in span]))  # [:e_cutoff]
    top_properties_ids = list(set([e_candidate['uri'] for e in top_properties.values() for e_candidate in e]))
    top_p_scores = {e_candidate['id']: e_candidate['score'] for e in top_properties.values() for e_candidate in e}
    n_e_activations = len(top_entities_ids)
    n_p_activations = len(top_properties_ids)

    # extract the subgraph
    kg = HDTDocument(hdt_path+hdt_file)
    kg.configure_hops(nhops, top_properties_ids, namespace, True)
    entities, predicate_ids, adjacencies = kg.compute_hops(top_entities_ids)
    kg.remove()

    # check if we hit the answer set
    correct_answers_ids = set(doc['answers_ids'])
    n_gs_answers = len(correct_answers_ids)
    n_hits = len(correct_answers_ids & set(entities))
    # accuracy
    acc = float(n_hits) / len(correct_answers_ids)
    accs.append(acc)

    # build adjacency matrix

    # index entity ids global -> local
    entities_dict = {k: v for v, k in enumerate(entities)}

    # generate a list of adjacency matrices per predicate assuming the graph is undirected wo self-loops
    adj_shape = (len(entities), len(entities))
    A = generate_adj_sp(adjacencies, adj_shape, include_inverse=True)

    # entity activation vector
    x = np.zeros(len(entities))
    for es in top_entities.values():
        # choose the first top entity per span
        for e in es:
            if e['id'] in entities_dict:
                x[entities_dict[e['id']]] = e['score']
                break

    # predicate activation vector
    p = np.zeros(len(predicate_ids))
    for es in top_properties.values():
        # choose the first top entity per span
        for e in es:
            if e['id'] in predicate_ids:
                p[predicate_ids.index(e['id'])] = e['score']
                break

    # 1 hop
    A1 = A * p

    # collect activations
    y1 = np.zeros(len(entities))
    activations1 = []
    for i, a_p in enumerate(A1):
        # activate current adjacency matrix via input propagation
        y_p = x @ a_p
        y1 += y_p

    # check output size
    assert y1.shape[0] == len(entities)

    # normalize activations by checking the 'must' constraints: number of constraints * weights
    # Y1 -= (len(q_ids) - 1) * 1

    # check activated entities
    top = np.argwhere(y1 == np.amax(y1)).T.tolist()[0]
    n_answers = len(top)

    # draw top activated entities from the distribution
    if n_answers:
        activations1 = np.asarray(entities)[top]

    # translate correct answers ids to local subgraph ids
    a_ids = [entities_dict[entity_id] for entity_id in correct_answers_ids if entity_id in entities_dict]
    
    n_correct = len(set(top) & set(a_ids))

    # report on error
    # if n_correct != n_gs_answers:
    #     print("!%d/%d"%(n_correct, n_gs_answers))

    # recall: answers that are correct / number of correct answers
    r = float(n_correct) / n_gs_answers

    if n_answers > 0:
        # precision: answers that are correct / number of answers
        p = float(n_correct) / n_answers
        # f-measure
        try:
            f = 2 * p * r / (p + r)
        except ZeroDivisionError:
            f = 0
    else:
        p = 0
        f = 0

    # add stats
    ps.append(p)
    rs.append(r)
    fs.append(f)

print("Acc: %.2f"%(np.mean(accs)))
print("P: %.2f R: %.2f F: %.2f"%(np.mean(ps), np.mean(rs), np.mean(fs)))
print("Fin. Results for %d questions"%len(ps))
