#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Jan 9, 2018

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Basic setup for KBQA experiments
'''

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
#                                                               "operator": "and",
                                                              "fields": ["label.snowball"],  # ["label.label", "label.ngrams"],  # , "label.ngrams" ,"label.snowball^50",  "label.snowball^20", "label.shingles",
                                                              }}},
                              size=top, doc_type=self.type)['hits']['hits']

    def label_scores(self, string, top=100, verbose=False):
        matches = self.es.search(index=self.index,
                              body={"query": {"multi_match": {"query": string,
#                                                               "operator": "and",
                                                              "fields": ["label.ngrams", "label.snowball"],  # ["label.label", "label.ngrams"],  # , "label.ngrams" ,"label.snowball^50",  "label.snowball^20", "label.shingles",
                                                              }}},
                              size=top, doc_type=self.type)['hits']
        span_ids = {}
        for match in matches['hits']:
            _id = match['_source']['id']
            score = match['_score'] / matches['max_score']
            span_ids[_id] = score
            if verbose:
              print({match['_source']['uri']: score})

        return span_ids

    def look_up_by_uri(self, uri, top=1):
        uri = uri.replace("'", "")
        results = self.es.search(index=self.index,
                              body={"query": {"term": {"uri": uri}}},
                              size=top, doc_type=self.type)['hits']['hits']
        if not results:
            results = self.es.search(index=self.index,
                              body={"query": {"term": {"uri": uri.replace("–", "-")}}},
                              size=top, doc_type=self.type)['hits']['hits']
            if not results:
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


# connect to MongoDB (27017 is the default port) to access the dataset
# sudo service mongod start 
from pymongo import MongoClient
import json

class Mongo_Connector():
    '''
    Wrapper class for some of the pymongo functions: http://api.mongodb.com/python/current/tutorial.html
    '''

    def __init__(self, db_name, col_name):
        # spin up database
        self.mongo_client = MongoClient()
        self.db = self.mongo_client[db_name]
        self.col = self.db[col_name]
    
    def count_all_docs(self):
        count = self.col.count_documents({})
        print ("%d docs"%count)
    
    def load_json(self, json_file_path):
        with open(json_file_path, "r") as json_file:
            docs = json.load(json_file)
        dataset_size = len(docs)
        print ("Inserting %d new docs"%(dataset_size))
        self.col.insert_many(docs)
        
    def show_sample(self):
        pprint.pprint(self.col.find_one())

    def get_all(self, limit=100):
        cursor = self.col.find({}, no_cursor_timeout=True)
        if limit:
            cursor = cursor.limit(limit)
        return cursor
        
    def get_sample(self, train=True, limit=100):
        '''
        Set limit to None to get all docs
        '''
        cursor = self.col.find({'train': train}, no_cursor_timeout=True).batch_size(1)
        if limit:
            cursor = cursor.limit(limit)
        return cursor
    
    def get_by_id(self, id, limit=1):
        '''
        '''
        cursor = self.col.find({'SerialNumber': id})
        if limit:
            cursor = cursor.limit(limit)
        return cursor


# load pre-trained word embeddings for question semantic representation
# wget http://magnitude.plasticity.ai/glove/medium/glove.840B.300d.magnitude
# wget https://public.ukp.informatik.tu-darmstadt.de/reimers/embeddings/komninos_english_embeddings.gz
# gunzip komninos_english_embeddings.gz | mv komninos_english_embeddings.txt
# python -m pymagnitude.converter -i komninos_english_embeddings.txt -o komninos_english_embeddings.magnitude
from pymagnitude import *

embeddings_path = "/home/zola/Projects/KBQA/data/embeddings/"

embeddings = {'glove6B100d': "glove.6B.100d.magnitude", 'glove840B300d': "glove.840B.300d.magnitude",
              'komninos': "komninos_english_embeddings.magnitude",
              'fasttext_p_labels': "predicates_labels_fasttext.magnitude",
              'fasttext_e_labels': "terms_labels_fasttext.magnitude"}


def load_embeddings(embeddings_choice):
    return Magnitude(embeddings_path+embeddings[embeddings_choice])
