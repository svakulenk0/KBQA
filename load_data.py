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


def load_lcquad_answers(split):
    '''
    Load answers for questions from LCQUAD dataset using SPARQL from an endpoint
    split <Str> train or test
    '''
    # prepare folder for the dataset
    if not os.path.exists(LCQUAD_DATASET_PATH):
        os.makedirs(LCQUAD_DATASET_PATH)

    questions_path = LCQUAD_DATASET_PATH + '/%s-data.json' % split

    # download questions dataset
    if not os.path.exists(questions_path):
        wget.download('https://raw.githubusercontent.com/AskNowQA/LC-QuAD/data/%s-data.json'%split, questions_path)

    with open(questions_path) as f:
        questions = json.load(f)

    with open("./data/lcquad_%s_all_answers.txt"%split, "w") as f:

        qas = []
        for question in questions:
            question_str = question["corrected_question"]
            print question_str
           
            # request the endpoint
            sparql_query = question["sparql_query"]
            # print sparql_query
            if "SELECT DISTINCT ?uri WHERE" in sparql_query:
                try:
                    response = requests.get(ENDPOINT, params={'query': sparql_query, 'output': 'json'})
                    results = response.json()['results']['bindings']
                    # print results
                    answers = [result.values()[0]['value'] for result in results if result.values()[0]['type'] == 'uri']
                    # print answers
                    # print '\n'
                    # return [path['X']['value'] for path in paths]
                except Exception, exc:
                    print exc
                
                if answers:
                    # if split == 'train':
                    #     f.write('<' + answers[0].encode('utf-8') + '>\n')
                    # elif split == 'test':
                    for answer in answers:
                        f.write('<' + answer.encode('utf-8') + '>\n')
                    qas.append({'question': question_str, 'answers': answers})

    with open("./data/lcquad_%s.json"%split, "w") as f:
        json.dump(qas, f, indent=2)


if __name__ == '__main__':
    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)
    # load_lcquad_answers('train')
    load_lcquad_answers('test')
