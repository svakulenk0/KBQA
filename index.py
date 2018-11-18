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

    def match_entities(self, query):
        results = self.es.search(index=self.index, body={"query": {"match": {"label": query}}}, doc_type=self.type)['hits']
        return results['hits']
        # if results['max_score']:
        #     if results['max_score'] > threshold:
        #         return results['hits'][0]
        # return None

    def parse_kb_uris(self, path="./data/entitiesWithObjectsURIs.txt"):

        with open(path, "r") as in_file:

            bulk_data = []

            for i, line in enumerate(in_file):
                # line template http://creativecommons.org/ns#license;2
                parse = line.split(';')
                entity_uri = ';'.join(parse[:-1])
                count = parse[-1].strip()
                entity_label = entity_uri.strip('/').split('/')[-1].strip('>').lower().strip('ns#')

                data_dict = {'uri': entity_uri, 'label': entity_label, 'count': count}

                op_dict = {
                    "index": {
                        "_index": self.index, 
                        "_type": self.type, 
                        "_id": i + 1
                    }
                }
                bulk_data.append(op_dict)
                bulk_data.append(data_dict)

            print len(bulk_data)
            return bulk_data

    def index_entities_bulk(self):
        '''
        Perform indexing 
        '''
        # create index
        self.build()

        # parse entities
        bulk_data = self.parse_kb_uris()

        # bulk index the data
        print("bulk indexing...")
        res = self.es.bulk(index=self.index, body=bulk_data, refresh=True)

        # sanity check
        res = self.es.search(index=self.index, size=2, body={"query": {"match_all": {}}})
        print(" response: '%s'" % (res))


def test_index_entities():
    es = IndexSearch()
    es.parse_kb_uris()
    es.index_entities_bulk()


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
    test_index_entities()
    # test_match_entities()
    # test_match_lcquad_questions()
