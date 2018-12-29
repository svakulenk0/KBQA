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
gs_annotations = True
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
                              body={"query": {"term": {"uri": quote(uri, safe='():/,')}}},
                              size=top, doc_type=self.type)['hits']['hits']
        return results

    def look_up_by_id(self, _id, top=1):
        results = self.es.search(index=self.index,
                              body={"query": {"term": {"id": _id}}},
                              size=top, doc_type=self.type)['hits']['hits']
        return results


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

if gs_annotations:
    annotation = ''
else:
    annotation = '_guess'

e_field, p_field = 'entity_ids%s'%annotation, 'predicate_uris%s'%annotation

samples = mongo.get_sample(limit=limit)

# subgraph extraction accuracy
accs = []
# hold macro-average stats for the model performance over the samples
ps, rs, fs = [], [], []

# iterate over the cursor
for doc in samples:
    # get correct entities and predicates from the GS annotations
    top_entities = doc[e_field]
    top_properties = doc[p_field]
    # extract the subgraph
    kg = HDTDocument(hdt_path+hdt_file)
    kg.configure_hops(nhops, top_properties, namespace, True)
    entities, predicate_ids, adjacencies = kg.compute_hops(top_entities)
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

    adj_shape = (len(entities), len(entities))
    # generate a list of adjacency matrices per predicate assuming the graph is undirected wo self-loops
    A = generate_adj_sp(adjacencies, adj_shape, include_inverse=True)
    # garbage collection
    del adjacencies

    # initial activations of entities
    # look up local entity id
    q_ids = [entities_dict[entity_id] for entity_id in top_entities if entity_id in entities_dict]
    # graph activation vector TODO activate with the scores
    X1 = np.zeros(len(entities))
    X1[q_ids] = 1
    print("%d question entities activated"%len(q_ids))

    # 1 hop
    # activate predicates for this hop
    p_activations = np.zeros(len(predicate_ids))
    # look up ids in index
    top_p_ids = []
    # TODO activate properties in top_properties
    for p_uri in top_properties:
      # if not in predicates check entities
      matches = p_index.look_up_by_uri(p_uri)
      if matches:
          top_p_ids.append(matches[0]['_source']['id'])
      else:
          print ("%s not found" % p_uri)

    p_ids = [i for i, p_id in enumerate(predicate_ids) if p_id in top_p_ids]

    # collect activations
    Y1 = np.zeros(len(entities))
    activations1 = []
    # slice A
    for a_p in A[p_ids]:
      # activate current adjacency matrix via input propagation
      y_p = X1 * a_p
      # check if there is any signal through
      if sum(y_p) > 0:
          # add up activations
          Y1 += y_p

    # normalize activations by checking the 'must' constraints: number of constraints * weights
    Y1 -= (len(q_ids) - 1) * 1

    # check activated entities
    top = np.argwhere(Y1 > 0).T.tolist()[0]
    n_answers = len(top)

    # draw top activated entities from the distribution
    if n_answers:
        activations1 = np.asarray(entities)[top]


    # translate correct answers ids to local subgraph ids
    a_ids = [entities_dict[entity_id] for entity_id in correct_answers_ids if entity_id in entities_dict]
    
    n_correct = len(set(top) & set(a_ids))

    # report on error
    if n_correct != n_gs_answers:
        print("!%d/%d"%(n_correct, n_gs_answers))

    # recall: answers that are correct / number of correct answers
    r = float(n_correct) / n_gs_answers

    if n_answers > 0:
        # precision: answers that are correct / number of answers
        p = float(n_correct) / n_answers
        # f-measure
        f = 2 * p * r / (p + r)
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
