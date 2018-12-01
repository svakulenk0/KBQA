#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Nov 26, 2018

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Crawl GS subgraph for each question from DBpedia HDT
'''
import numpy as np
from subprocess import call, Popen, PIPE
import scipy.sparse as sp

from lcquad import load_lcquad
from index import IndexSearch


limit = 1

# get a random sample of questions from lcquad train split
samples = load_lcquad(fields=['corrected_question', 'entities', 'answers'], dataset_split='train',
                      shuffled=True, limit=limit)

es = IndexSearch('dbpedia2016-04')
# interate over questions entities dataset
for question, correct_question_entities, answers in samples:
    print (question)
    print (correct_question_entities)

    # pick a seed entity for each question
    matched_uris = []
    question_entity_ids = []
    # add answer entities
    # correct_question_entities.extend(answers)
    for entity_uri in correct_question_entities:
        matches = es.match_entities(entity_uri, match_by='uri')
        # p
        if len(matches) > 1:
            for match in matches:
                if match['_source']['term_type'] == "predicates":
                    question_entity_ids.append(match['_source']['id'])
        # s o
        elif matches:
            matched_uris.append(matches[0]['_source']['uri'])
            question_entity_ids.append(matches[0]['_source']['id'])
    # print (matched_uris)
    print (question_entity_ids)

    # get id of the answer entities
    answer_entity_ids = []
    for entity_uri in answers:
        matches = es.match_entities(entity_uri, match_by='uri')
        for match in matches:
            answer_entity_ids.append(matches[0]['_source']['id'])
    print (answer_entity_ids)

    # request subgraph from the API (2 hops from the seed entity)
    hdt_lib_path = "/home/zola/Projects/hdt-cpp-molecules/libhdt"
    p = Popen(["/home/zola/Projects/hdt-cpp-molecules/libhdt/tools/hops", "-t", "<%s>"%matched_uris[0], '-p', "http://dbpedia.org/", '-n', '2', 'data/dbpedia2016-04en.hdt'], stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=hdt_lib_path)
    subgraph, err = p.communicate()
    print (subgraph)
