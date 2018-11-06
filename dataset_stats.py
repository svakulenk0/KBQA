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


def check_qa_entities_in_kb(dataset_split='train'):
    '''
    Check how many questions can be answered with the existing set of KG embeddings
    '''
    # get entity URis
    with open("./data/DBpedia_KGlove_uris.txt", "r") as file:
        entity_uris = file.read().splitlines()
        print ("Loaded %d entity uris"%len(entity_uris))
        print ("Unique %d entity uris"%len(set(entity_uris)))

    # init counter
    n_select_questions = 0
    n_select_questions_answerable = 0
    # load QA dataset
    with open("./data/lcquad_%s_new.json"%dataset_split, "r") as train_file:
        qas = json.load(train_file)
        print ("%d total QA pairs in lcquad %s" % (len(qas), dataset_split))
        for qa in qas:
            # skip bool and int queries
            sparql_query = qa["sparql_query"]
            # flag
            all_entities_found = True
            if "SELECT DISTINCT ?uri WHERE" in sparql_query:
                n_select_questions += 1
                # make sure that all entities are in the kg 
                entities = qa['entities']
                # print entities
                for entity in entities:
                    entity = entity.encode('utf-8')
                    if entity not in entity_uris:
                        all_entities_found = False
                        break
                if all_entities_found:
                    # make sure that at least one entity is the kg 
                    answers = qa['answers']
                    # print answers
                    for answer in answers:
                        answer = answer.encode('utf-8')
                        if answer in entity_uris:
                            n_select_questions_answerable += 1
                            break


if __name__ == '__main__':
    check_qa_entities_in_kb()
