#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Aug 24, 2018

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Load answers for questions from datasets using SPARQL against an endpoint

'''
import os
import wget
import json
import requests

DATASET_PATH = "./data/"
LCQUAD_DATASET_PATH = "./data/lcquad"
ENDPOINT = 'http://hdt.communidata.at/dbpedia/query'


def load_dbnqa_answers():
    # https://ndownloader.figshare.com/articles/6118505/versions/2
    # TODO: decode special symbols
    # which DBpedia version?
    pass


def load_lcquad_answers():
    '''
    Load answers for questions from LCQUAD dataset using SPARQL from an endpoint
    '''
    training_questions_path = LCQUAD_DATASET_PATH+'/train-data.json'
    
    # download questions dataset
    if not os.path.exists(LCQUAD_DATASET_PATH):
        os.makedirs(LCQUAD_DATASET_PATH)
        wget.download('https://raw.githubusercontent.com/AskNowQA/LC-QuAD/data/train-data.json', training_questions_path)

    with open(training_questions_path) as f:
        questions = json.load(f)

    with open("./data/lcquad_train.json", "w") as write_file:

        for question in questions:
            question_str = question["corrected_question"]
            
            sparql_query = question["sparql_query"]
            # filter only select queries
            if "SELECT DISTINCT ?uri WHERE" in sparql_query:
                print question_str
                print sparql_query
                # request the SPARQL endpoint
                try:
                    response = requests.get(ENDPOINT, params={'query': sparql_query, 'output': 'json'})
                    results = response.json()['results']['bindings']
                    # print results
                    answers = [result.values()[0]['value'] for result in results]
                    # return [path['X']['value'] for path in paths]
                except Exception, exc:
                    print exc

                if answers:
                    print answers
                    print '\n'
                    qa = {'question': question_str, 'answers': answers}
                    json.dump(qa, write_file, indent=2)


if __name__ == '__main__':
    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)
    load_lcquad_answers()
