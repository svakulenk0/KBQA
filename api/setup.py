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

    def label_scores(self, string, top=100, verbose=False, threshold=1.0, scale=None, max_degree=None):
        matches = self.es.search(index=self.index,
                              body={"query": {"multi_match": {"query": string,
#                                                               "operator": "and",
                                                              "fields": ["label.ngrams", "label.snowball^20"],  # ["label.label", "label.ngrams"],  # , "label.ngrams" ,"label.snowball^50",  "label.snowball^20", "label.shingles",
                                                              }}},
                              size=top, doc_type=self.type)['hits']
        span_ids = {}
        for match in matches['hits']:
            _id = match['_source']['id']
            degree = int(match['_source']['count'])
            if max_degree and degree <= max_degree:
              score = match['_score'] / matches['max_score']
              if not threshold or score >= threshold:
                  if scale:
                    score *= scale
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


# MP functions
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


def hop(entities, constraints, top_predicates, verbose=False, max_triples=500000, bl_p=[68655]):
    '''
    Extract the subgraph for the selected entities
    bl_p  -- the list of predicates to ignore (e.g. type predicate is too expensive to expand)
    ''' 
#     print(top_predicates)
    n_constraints = len(constraints)
    if entities:
        n_constraints += 1

    top_entities = entities + constraints
    all_entities_ids = [_id for e in top_entities for _id in e]
    top_predicates_ids = [_id for p in top_predicates for _id in p if _id not in bl_p]

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
