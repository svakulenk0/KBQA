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

LCQUAD_DATASET_PATH = "lcquad_clean.json"  # wget https://raw.githubusercontent.com/AskNowQA/EARL/master/data/lcquad.json
ENDPOINT = 'http://localhost:8164/sparql'
ns_filter = "http://dbpedia.org/"  # process only entities with URIs from the DBpedia namespace


def load_lcquad_answers(save=True):
    '''
    Load answers for questions from LCQUAD dataset using SPARQL endpoint
    '''
    with open(LCQUAD_DATASET_PATH, encoding='utf-8') as f:
        questions = json.load(f)
    dataset = []
    for question in questions:
        print(question['question'])
        answers = []
        # fix queries
        sparql_query = question["sparql_query"]  # .replace("'", "")
        if "SELECT DISTINCT COUNT" in sparql_query:
            sparql_query = sparql_query.replace("SELECT DISTINCT COUNT(?uri)", "SELECT DISTINCT ?uri")
            question['question_type'] = 'COUNT'
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
            question['answers'] = answers
            if not answers:
                # show missing answers
                print(sparql_query)
            if 'question_type' not in question:
                question['question_type'] = 'SELECT'

        elif "ASK WHERE" in sparql_query:
            question['bool_answer'] = response['boolean']
            question['question_type'] = 'ASK'
        else:
            # unexpected query type
            print(response)
        if question['question_type'] == 'COUNT':
            question['count_answer'] = len(question['answers'])

        # store the QA sample
        dataset.append(question)

    print("%d QA samples"%len(dataset))
    # save the dataset with answers
    with open("lcquad_answers.json", "w") as f:
        json.dump(dataset, f, indent=2)


if __name__ == '__main__':
    load_lcquad_answers()
