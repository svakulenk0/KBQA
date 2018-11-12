#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Nov 6, 2018

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Check datasets statistics
'''
import json

EMBEDDINGS_PATH = "./data/embeddings/"
KB_RELATION_EMBEDDINGS_PATH = EMBEDDINGS_PATH + 'DBpediaVecotrs200_20Shuffle.txt'


def count_QA_entities(dataset_split='train'):
    # load QA dataset
    with open("./data/lcquad_%s_new.json"%dataset_split, "r") as train_file, open("./data/lcquad_%s_new_entities.txt"%dataset_split, "w") as out_file:
        qas = json.load(train_file)
        print ("%d total QA pairs in lcquad %s" % (len(qas), dataset_split))
        for qa in qas:
            # skip bool and int queries
            sparql_query = qa["sparql_query"]
            # print sparql_query
            if "SELECT DISTINCT ?uri WHERE" in sparql_query:
                answers = qa['answers']
                print answers
                out_file.write('\n'.join([answer.encode('utf-8')for answer in answers]) + '\n')
                entities = qa['entities']
                print entities
                out_file.write('\n'.join([entity.encode('utf-8')for entity in entities]) + '\n')
                # return


def check_qa_entities_in_kb(dataset_split='train', path_kg_uris="./entitiesWithObjects_uris.txt"):
    '''
    Check how many questions can be answered with the existing set of KG entities
    '''
    path_qa_dataset = "./lcquad_%s_new.json"%dataset_split

    entities = []
    with open(path_qa_dataset, "r") as file:
        qas = json.load(file)
        print ("%d total QA pairs in lcquad %s" % (len(qas), dataset_split))
        for qa in qas:
            sparql_query = qa["sparql_query"]
            # skip bool and int queries
            if "SELECT DISTINCT ?uri WHERE" in sparql_query:
                if qa['answers']:
                    entities.extend(qa['entities'])
                    entities.extend(qa['answers'])
    entities = set(entities)
    print ("%d entities in lcquad %s" % (len(entities), dataset_split))

    # load QA dataset
    with open(path_kg_uris, "r") as file:
        for entity in file:
            entity = entity.strip('\n')
            if entity in entities:
                entities.remove(entity)

    print ("%d entities left unmatched in lcquad %s" % (len(entities), dataset_split))


if __name__ == '__main__':
    check_qa_entities_in_kb()
