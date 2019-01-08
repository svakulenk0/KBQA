#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Jan 8, 2018

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Load all lcquad answers
'''

import os
import json
import requests

LCQUAD_DATASET_PATH = "lcquad/lcquad.json"
ENDPOINT = 'http://wikidata.communidata.at/dbpedia/query'
ns_filter = "http://dbpedia.org/"  # process only entities with URIs from the DBpedia namespace


def load_lcquad_answers(save=True):
    '''
    Load answers for questions from LCQUAD dataset using SPARQL endpoint
    '''
    with open(LCQUAD_DATASET_PATH, encoding='utf-8') as f:
        questions = json.load(f)
    dataset = []
    for question in questions:
        # print(question['question'])
        answers = []
        # fix URI
        sparql_query = question["sparql_query"].replace('https://www.w3.org', 'http://www.w3.org').replace("'", "").replace("?uri, ?x", "?uri")
        sparql_query = sparql_query.replace("SELECT DISTINCT COUNT(?uri)", "SELECT (COUNT (DISTINCT ?uri) as ?count)")
        # request the endpoint
        response = requests.get(ENDPOINT, params={'query': sparql_query, 'output': 'json'}).json()
        if "SELECT DISTINCT ?uri WHERE" in sparql_query and 'results' in response:
            results = response['results']['bindings']
            for result in results:
                if result:
                    # retain only URIs as answers
                    if 'uri' in result.keys():
                        if result['uri']['type'] == 'uri':
                            answer = result['uri']['value']
                            # consider only DBpedia URIs as answers
                            if answer.startswith(ns_filter):
                                answers.append(answer)
        elif "ASK WHERE" in sparql_query:
            answers = response['boolean']
        elif "SELECT (COUNT" in sparql_query:
            answers = response['results']['bindings'][0]['count']['value']
        else:
            print(response)

        if answers:
            question['answers'] = answers
            dataset.append(question)
        else:
            # show missing SELECT query answers
            print(sparql_query)
    
    print("%d queries with answers found"%len(dataset))
    # save the dataset with answers
    with open("../data/lcquad_answers.json", "w") as f:
        json.dump(dataset, f, indent=2)


if __name__ == '__main__':
    load_lcquad_answers()
