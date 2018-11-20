#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Nov 15, 2018

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Index entities into ES

Following https://qbox.io/blog/building-an-elasticsearch-index-with-python
'''
import io
import re

from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk


class IndexSearch:
    
    def __init__(self):
        # set up ES connection
        self.es = Elasticsearch()
        self.index = 'dbpedia2016-04'
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

    def match_entities(self, query=None, match_by="label"):
        if query:
            results = self.es.search(index=self.index, body={"query": {"match": {match_by: query}}}, doc_type=self.type)['hits']
        else:
            results = self.es.search(index=self.index, size=2, body={"query": {"match_all": {}}})['hits']

        return results['hits']

        # if results['max_score']:
        #     if results['max_score'] > threshold:
        #         return results['hits'][0]
        # return None
        # sanity check

    def uris_stream(self, file_to_index_path):

        with io.open(file_to_index_path, "r", encoding='utf-8') as file:
            for i, line in enumerate(file):
                # print(line)
                # line template http://creativecommons.org/ns#license;2
                # filter out only dbpedia URIs
                dbpedia_uri = re.search("^http://dbpedia.org", line)
                if dbpedia_uri:
                    # print (line)
                    parse = line.split(';')
                    entity_uri = ';'.join(parse[:-1])
                    count = parse[-1].strip()
                    entity_label = entity_uri.strip('/').split('/')[-1].strip('>').lower().strip('ns#')

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

        for ok, response in streaming_bulk(self.es, actions=self.uris_stream(file_to_index), chunk_size=100000):
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


def test_match_lcquad_questions(limit=10, check_uri_exist=False):
    '''
    Estimate entity linking performance (candidate selection) via ES index
    '''
    es = IndexSearch()

    import pickle
    from keras.preprocessing.text import text_to_word_sequence
    from lcquad import load_lcquad

    wfd = pickle.load(open("wfd.pkl", "rb"))

    # get a random sample of questions from lcquad train split
    samples = load_lcquad(fields=['corrected_question', 'entities'], dataset_split='train',
                          shuffled=True, limit=limit)

    # iterate over questions and check for how many questions we can hit the correct entity set
    hits = 0
    # samples = [["what is the fuel capacity", ["http://dbpedia.org/ontology/Automobile/fuelCapacity"]]]
    for question, correct_question_entities in samples:
        # show sample
        # print(question)
        # print(correct_question_entities)

        if check_uri_exist:
            # check that we have all the entities referenced in the question
            matched_uris = [match['_source']['uri'] for entity_uri in correct_question_entities for match in es.match_entities(entity_uri, match_by='uri')]
            print(matched_uris)

            # check them against correct uris and filter out only the correctly matched URIs
            correct_matched_uris = [matched_uri for matched_uri in matched_uris if matched_uri in correct_question_entities]
            print(correct_matched_uris)

            # consider a hit if we managed to match at least one correct URI
            if correct_matched_uris:
                hits += 1

        # select words to look up in ES
        # selected_words = [word for word in question.split()]
        # selected_words = [word for word in text_to_word_sequence(question, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\'')]
        selected_words = [word for word in text_to_word_sequence(question, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\'') if word not in wfd.keys()]
        # print(selected_words)

        # look up all relevant URIs for the selected words
        matched_uris = [match['_source']['uri'] for word in selected_words for match in es.match_entities(word, match_by='label')]
        # print(matched_uris)

        # check them against correct uris and filter out only the correctly matched URIs
        correct_matched_uris = [matched_uri for matched_uri in matched_uris if matched_uri in correct_question_entities]
        # print(correct_matched_uris)

        # consider a hit if we managed to match at least one correct URI
        if correct_matched_uris:
            hits += 1
        # else:
        #     # report case
        #     print(question)
        #     print(correct_question_entities)
        #     print(matched_uris)

    print ("%d hits out of %d"%(hits, len(samples)))


if __name__ == '__main__':
    # test_index_entities()
    # test_match_entities()
    test_match_lcquad_questions()
