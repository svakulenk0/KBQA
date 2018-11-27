#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Nov 26, 2018

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Crawl GS subgraph for each question from DBpedia HDT
'''
from subprocess import call, Popen, PIPE

from lcquad import load_lcquad
from index import IndexSearch


limit = 10

# get a random sample of questions from lcquad train split
samples = load_lcquad(fields=['corrected_question', 'entities'], dataset_split='train',
                      shuffled=True, limit=limit)

es = IndexSearch()
# interate over questions entities dataset
for question, correct_question_entities in samples:
    print (correct_question_entities)
    # pick a seed entity for each question
    matched_uris = [match['_source']['uri'] for entity_uri in correct_question_entities for match in es.match_entities(entity_uri, match_by='uri') if match['_source']['term_type'] != "predicates"]
    print (matched_uris)

    # request subgraph from the API (2 hops from the seed entity)
    # /home/zola/Projects/hdt-cpp-molecules/libhdt/tools/hops -t "<http://dbpedia.org/resource/David_King-Wood>" -p "http://dbpedia.org/" -n 2 /home/zola/Projects/hdt-cpp-molecules/libhdt/data/dbpedia2016-04en.hdt
    hdt_lib_path = "/home/zola/Projects/hdt-cpp-molecules/libhdt"
    # call([hdt_lib_path%"tools/hops", "-t", matched_uris[0], '-p', "http://dbpedia.org/", '-n', '2', '-o', 'result.txt', hdt_lib_path%'data/dbpedia2016-04en.hdt'])
    p = Popen(["/home/zola/Projects/hdt-cpp-molecules/libhdt/tools/hops", "-t", matched_uris[0], '-p', "http://dbpedia.org/", '-n', '2', '-o', 'result.txt', 'data/dbpedia2016-04en.hdt'], stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=hdt_lib_path)
    out, err = p.communicate()
    print (p)
    print (out)

    # while True:
    #   line = p.stdout.readline()
    #   if line != '':
    #     print (line.rstrip())
    #   else:
    #     break

    # verify subgraph, i.e. all question entities are within the extracted subgraph

    # store subgraph
