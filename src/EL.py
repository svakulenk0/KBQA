#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Jan 8, 2018

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Evaluate entity linking performance and store annotations
'''

# setup
dataset_name = 'lcquad'

import os
os.chdir('/home/zola/Projects/KBQA/src')

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
                                                              "fields": ["label^10", "label.ngrams"],  # ,"label.snowball^50",  "label.snowball^20", "label.shingles",
                                                              }}},
                              size=top, doc_type=self.type)['hits']['hits']

    def look_up_by_uri(self, uri, top=1):
        results = self.es.search(index=self.index,
                              body={"query": {"term": {"uri": quote(uri, safe=string.punctuation)}}},
                              size=top, doc_type=self.type)['hits']['hits']
        return results

    def look_up_by_id(self, _id, top=1):
        results = self.es.search(index=self.index,
                              body={"query": {"term": {"id": _id}}},
                              size=top, doc_type=self.type)['hits']['hits']
        return results

    def look_up_by_label(self, _id):
        results = self.es.search(index=self.index,
                                 body={"query": {"term": {"label_exact": _id}}},
                                 doc_type=self.type)['hits']['hits']
        return results


e_index = IndexSearch('dbpedia201604e')

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
        cursor = self.col.find({'train': True})
        if limit:
            cursor = cursor.limit(limit)
        return cursor


mongo = Mongo_Connector('kbqa', dataset_name)

# load embeddings
from pymagnitude import *
embeddings_path = "/home/zola/Projects/KBQA/data/embeddings/"
embeddings = {'fasttext_p_labels': "predicates_labels_fasttext.magnitude",
              'fasttext_e_labels': "terms_labels_fasttext.magnitude"}
e_vectors = Magnitude(embeddings_path+embeddings['fasttext_e_labels'])

# match and save matched entity URIs to MongoDB TODO evaluate against correct spans
limit = None
string_cutoff = 100  # maximum number of candidate entities per mention
semantic_cutoff = 20

# path to KG relations
from hdt import HDTDocument
hdt_path = "/home/zola/Projects/hdt-cpp-molecules/libhdt/data/"
hdt_file = 'dbpedia2016-04en.hdt'
namespace = "http://dbpedia.org/"

import numpy as np
print("Evaluating entity linking...")
def entity_linking(spans_field, save, show_errors=True, add_nieghbours=True):
    # iterate over the cursor
    samples = mongo.get_sample(limit=limit)
    count = 0
    # hold macro-average stats for the model performance over the samples
    ps, rs, fs = [], [], []
    for doc in samples:
        correct_uris = doc['entity_uris']
        print(set(correct_uris))
        # get entity spans
        e_spans = doc[spans_field+'_guess']
    #     print(e_spans)
        # get entity matches TODO save scores
        top_ids = []
        top_entities = {}
        for span in e_spans:
            print("Span: %s"%span)
            print("Index lookup..")
            guessed_labels, guessed_ids = [], []
            for match in e_index.match_label(span, top=string_cutoff):
                label = match['_source']['label_exact']
                if label not in guessed_labels:
                    guessed_labels.append(label)
                # avoid expanding heavy hitters
                degree = match['_source']['count']
                if int(degree) < 100000:
                    guessed_ids.append(match['_source']['id'])
            
            print("%d candidate labels"%len(guessed_labels))
            if add_nieghbours:
                print("KG lookup..")
                kg = HDTDocument(hdt_path+hdt_file)
                kg.configure_hops(1, [], namespace, True)
                entities, predicate_ids, adjacencies = kg.compute_hops(guessed_ids)
                kg.remove()
                # look up labels
                for e_id in entities:
                    match = e_index.look_up_by_id(e_id)
                    if match:
                        label = match[0]['_source']['label_exact']
                        if label not in guessed_labels:
                            guessed_labels.append(label)
                guessed_ids.extend(entities)
            
            # remove duplicates
#             guessed_labels = list(set(guessed_labels))
            
            # score with embeddings
            top_labels = []
            print("%d candidate labels"%len(guessed_labels))
#             for candidate in guessed_labels:
#                 if candidate not in e_vectors:
#                     print(candidate)
            print("Embeddings lookup..")
            dists = e_vectors.distance(span, [label for label in guessed_labels if label in e_vectors])
            top = np.argsort(dists)[:semantic_cutoff].tolist()
            top_labels = [guessed_labels[i] for i in top]
        
            print("selected labels: %s"%top_labels)
            print("Index lookup..")
            top_entities[span] = []
            for i, label in enumerate(top_labels):
                print(label)
                for match in e_index.look_up_by_label(label):
                    distance = float(dists[top[i]])
                    degree = match['_source']['count']
                    _id = match['_source']['id']
                    uri = match['_source']['uri']
                    print(uri)
                    top_entities[span].append({'rank': i+1, 'distance': distance, 'degree': degree, 'id': _id, 'uri': uri})
                    top_ids.append(_id)
        print(top_entities)
            
        # evaluate against the correct entity ids
        top_ids = list(set(top_ids))
        correct_ids = set(doc['entity_ids'])
        n_hits = len(correct_ids & set(top_ids))
        try:
            r = float(n_hits) / len(correct_ids)
        except ZeroDivisionError:\
            print(doc['question'])
        try:
            p = float(n_hits) / len(top_ids)
        except ZeroDivisionError:
            p = 0
        try:
            f = 2 * p * r / (p + r)
        except ZeroDivisionError:
            f = 0

        # add stats
        ps.append(p)
        rs.append(r)
        fs.append(f)

        # save to MongoDB
        if save:
            doc[spans_field+'_guess'] = top_entities
            mongo.col.update_one({'_id': doc['_id']}, {"$set": doc}, upsert=True)
            count += 1

    print("P: %.2f R: %.2f F: %.2f"%(np.mean(ps), np.mean(rs), np.mean(fs)))
    print("Fin. Results for %d questions"%len(ps))    
    if save:
        print("%d documents annotated with entity ids guess"%count)

    

# evaluate entity linking on extracted entity spans
entity_linking(spans_field='entity_spans', save=True)
