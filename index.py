#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Nov 15, 2018

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Index entities into ES

Following https://qbox.io/blog/building-an-elasticsearch-index-with-python
'''

from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk


class IndexSearch:
    
    def __init__(self):
        # set up ES connection
        self.es = Elasticsearch()
        self.index = 'dbpedia201604'
        self.type = 'entities'

    def build(self):
        '''
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
        res = self.es.indices.create(index=self.index, body=request_body)
        print(" response: '%s'" % (res))

    def match_entities(self, query=None):
        if query:
            results = self.es.search(index=self.index, body={"query": {"match": {"label": query}}}, doc_type=self.type)['hits']
        else:
            results = self.es.search(index=self.index, size=2, body={"query": {"match_all": {}}})['hits']

        return results['hits']

        # if results['max_score']:
        #     if results['max_score'] > threshold:
        #         return results['hits'][0]
        # return None
        # sanity check

    def uris_stream(self, file_to_index_path):

        with open(file_to_index_path, "rb") as file:
            for i, line in enumerate(file):
                # print(line)
                # line template http://creativecommons.org/ns#license;2
                parse = line.split(';')
                entity_uri = ';'.join(parse[:-1]).encode('utf-8')
                count = int(parse[-1].strip())
                entity_label = entity_uri.strip('/').split('/')[-1].strip('>').lower().strip('ns#').encode('utf-8')

                data_dict = {'uri': entity_uri, 'label': entity_label, 'count': count, "id": i + 1}

                yield {"_index": self.index,
                       "_type": self.type,
                       "_source": data_dict
                       }

    def index_entities_bulk(self, file_to_index="./data/entitiesWithObjectsURIs.txt"):
        '''
        Perform indexing
        https://www.elastic.co/guide/en/elasticsearch/reference/current/tune-for-indexing-speed.html
        '''
        # create index
        self.build()

        # iterate via streaming_bulk following https://stackoverflow.com/questions/34659198/how-to-use-elasticsearch-helpers-streaming-bulk
        print("bulk indexing...")

        for ok, response in streaming_bulk(self.es, actions=self.uris_stream(file_to_index), chunk_size=500):
            if not ok:
                # failure inserting
                print (response)


def test_index_entities():
    es = IndexSearch()
    es.index_entities_bulk()


def test_match_entities():
    es = IndexSearch()
    
    print (es.match_entities())

    query = 'license'
    print (es.match_entities(query))


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
                print (question)
                print (word)
                print (es.match_entities(word))

        if i == limit:
            break


if __name__ == '__main__':
    test_index_entities()

    # test_match_entities()

    # test_match_lcquad_questions()
