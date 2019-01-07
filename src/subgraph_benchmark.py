#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Dec 28, 2018

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Evaluate subgraph extraction
'''
import numpy as np
from pymongo import MongoClient
from hdt import HDTDocument
from elasticsearch import Elasticsearch
from urllib.parse import quote

# setup
dataset_name = 'lcquad'

# connect to MongoDB (27017 is the default port) to access the dataset
# sudo service mongod start 
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
hdt_path = "/home/zola/Projects/hdt-cpp-molecules/libhdt/data/"
hdt_file = 'dbpedia2016-04en.hdt'
namespace = "http://dbpedia.org/"

# connect to entity catalog

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
        if not results:
            # fall back to label match
            return self.match_label(uri.split('/')[-1], top=1)
            
        return results


e_index = IndexSearch('dbpedia201604e')

# get the answer set hits

def evaluate_subgraph_extraction(nhops, e_field, p_field, limit=None, show_errors=False):
    '''
    e_field, p_field <str> names of the fields in MongoDB to look up the IDs
    '''
    samples = mongo.get_sample(limit=limit)
    # iterate over the cursor
    accs = []
    for doc in samples:
        # get correct entities and predicates from the GS annotations
        e_ids = doc[e_field]
        p_uris = doc[p_field]

        # extract the subgraph
        kg = HDTDocument(hdt_path+hdt_file)
        kg.configure_hops(nhops, p_uris, namespace, True)
        entities, _, _ = kg.compute_hops(e_ids)
        kg.remove()
        
        # check if we hit the answer set
        if 'answers_ids' in doc:
            correct_answers_ids = set(doc['answers_ids'])
    #         print(correct_answers_ids)
            n_hits = len(correct_answers_ids & set(entities))
            # accuracy
            acc = float(n_hits) / len(correct_answers_ids)
            accs.append(acc)
            if show_errors & (acc < 1):
                print(doc['question'])
                print(doc['entity_ids'])
                print(doc['predicate_uris'])
    return accs

# 1-hop subgraphs
# accs1 = evaluate_subgraph_extraction(nhops=1, e_field='entity_ids', p_field='predicate_uris_guess')
# print("Acc: %.2f"%np.mean(accs1))

# accs1 = evaluate_subgraph_extraction(nhops=1, e_field='entity_ids_guess', p_field='predicate_uris')
# print("Acc: %.2f"%np.mean(accs1))

# accs1 = evaluate_subgraph_extraction(nhops=1, e_field='entity_ids_guess', p_field='predicate_uris_guess')
# print("Acc: %.2f"%np.mean(accs1))

# 2-hop subgraphs
accs2 = evaluate_subgraph_extraction(nhops=2, e_field='entity_ids', p_field='predicate_uris')
print("Acc: %.2f"%np.mean(accs2))

# accs2 = evaluate_subgraph_extraction(nhops=2, e_field='entity_ids', p_field='predicate_uris_guess')
# print("Acc: %.2f"%np.mean(accs2))

# accs2 = evaluate_subgraph_extraction(nhops=2, e_field='entity_ids_guess', p_field='predicate_uris')
# print("Acc: %.2f"%np.mean(accs2))

# accs2 = evaluate_subgraph_extraction(nhops=2, e_field='entity_ids_guess', p_field='predicate_uris_guess')
# print("Acc: %.2f"%np.mean(accs2))
