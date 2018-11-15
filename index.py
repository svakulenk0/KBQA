#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Nov 15, 2018

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Index entities into ES
'''

from elasticsearch import Elasticsearch


class IndexSearch:
    
    def __init__(self):
        # set up ES connection
        self.es = Elasticsearch()
        self.index = 'dbpedia2016-04'
        self.type = 'entities'

    def build(self):
        '''
        following https://qbox.io/blog/building-an-elasticsearch-index-with-python
        '''
        if self.es.indices.exists(self.index):
            print("deleting '%s' index..." % (self.index))
            res = self.es.indices.delete(index = self.index)
            print(" response: '%s'" % (res))
        # since we are running locally, use one shard and no replicas
        request_body = {
            "settings" : {
                "number_of_shards": 1,
                "number_of_replicas": 0
            }
        }
        print("creating '%s' index..." % (self.index))
        res = self.es.indices.create(index = self.index, body = request_body)
        print(" response: '%s'" % (res))

    def match_entities(self, query):
        results = self.es.search(index=self.index, body={"query": {"match": {"label": query}}}, doc_type=self.type)['hits']
        return results['hits']
        # if results['max_score']:
        #     if results['max_score'] > threshold:
        #         return results['hits'][0]
        # return None

    def index_entities(self, path="./data/entitiesWithObjectsURIs.txt"):
        '''
        Perform indexing 
        '''
        self.build()
        with open(path, "r") as infile:
            for line in infile:
                # line template http://creativecommons.org/ns#license;2
                parse = line.split(';')
                entity_uri = ';'.join(parse[:-1])
                count = parse[-1].strip()
                entity_label = entity_uri.strip('/').split('/')[-1].strip('>').lower().strip('ns#')

                self.es.index(index=self.index, doc_type=self.type,
                              body={'id': entity_uri, 'label': entity_label, 'count': count})

        # sanity check
        res = self.es.search(index = self.index, size=2, body={"query": {"match_all": {}}})
        print(" response: '%s'" % (res))


def test_index_entities():
    es = IndexSearch()
    es.index_entities()


def test_match_entities():
    es = IndexSearch()
    query = 'license'
    print es.match_entities(query)


def test_match_lcquad_questions(limit=10):
    es = IndexSearch()

    import pickle
    from keras.preprocessing.text import text_to_word_sequence
    from lcquad import load_lcquad_qe

    wfd = pickle.load( open( "wfd.pkl", "rb" ) )

    questions, correct_question_entities = load_lcquad_qe()

    # iterate over questions
    for i, question in enumerate(questions):
        for word in text_to_word_sequence(question, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\''):
            if word in wfd.keys():
                # print(wfd[word])
                pass
            else:
                print question
                print word
                print es.match_entities(word)

        if i == limit:
            break


if __name__ == '__main__':
    # test_index_entities()
    # test_match_entities()
    test_match_lcquad_questions()
