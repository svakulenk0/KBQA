#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Nov 4, 2018

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Evaluate question entity selection using word embeddings
'''
import io
import json

from keras.preprocessing.text import text_to_word_sequence
from pymagnitude import *

KGLOVE_PATH = './data/embeddings/DBpediaVecotrs200_20Shuffle.txt'

test_q = "How many movies did Stanley Kubrick direct?"
kg_entities_path = './data/lcquad_train_entities.txt'
correct_entities = ["http://dbpedia.org/ontology/director", "http://dbpedia.org/resource/Stanley_Kubrick"]


def save_words_list(words_list, file_name):
    with open(file_name, 'w') as out:
        out.write('\n'.join(words_list) + '\n')


def produce_word_list(kg_embeddings_path=KGLOVE_PATH, out_file_path='./data/DBpedia_KGlove_labels.txt'):
    # parse entity embeddings file
    with open(kg_embeddings_path) as embeddings_file, open(out_file_path, 'w') as out:
        # iterate over lines (glove format)
        for line in embeddings_file:
            record = line.strip().split()
            entity_uri = record[0]
            # strip the domain name from the entity_uri, brakets and category: prefix to produce a cleaner entity label
            entity_label = entity_uri.strip('\n').strip('/').strip('>').split('/')[-1].split(':')[-1]
            print entity_label
            out.write(entity_label + '\n')


def load_lcquad(dataset_split='train'):
    QS = []
    ES = []
    empty_answers = 0
    with open("./data/lcquad_%s_new.json"%dataset_split, "r") as train_file:
        qas = json.load(train_file)
        
        print ("%d total QA pairs in lcquad %s" % (len(qas), dataset_split))

        for qa in qas:
            if qa['answers']:
                QS.append(qa['corrected_question'])
                ES.append(qa['entities'])
            else:
                empty_answers += 1
    
    print ("%d questions skipped because no answer was found" % empty_answers)
    return (QS, ES)


def test_embeddings(fname_kg='./data/lcquad_train_entities_labels_fasttext.magnitude'):
    
    questions, correct_question_entities = load_lcquad()

    # shuffle samples
    # TODO

    # load embeddings
    kg_word_embeddings = Magnitude(fname_kg)
    # sample
    n_samples = 2

    # keep counters
    i = 0
    n_entities = 0
    n_entities_kg = 0
    hits = 0

    # iterate over questions
    for question in questions:
        # preprocess entities
        correct_question_entity_labels = [entity_uri.strip('\n').strip('/').strip('>').split('/')[-1]for entity_uri in correct_question_entities[i]]
        n_entities += len(correct_question_entity_labels)

        # check that correct entities are in our KG
        existing_correct_entities = [entity for entity in correct_question_entity_labels if entity in kg_word_embeddings]
        n_entities_kg = len(existing_correct_entities)
        print("%d out of %d entities required to answer the question found in the KG"%(n_entities_kg, n_entities))

        candidates = []
        
        for word in text_to_word_sequence(question):
            # print(word)
            # top = kg_word_embeddings.most_similar(word, topn=100) # Most similar by key
            top = kg_word_embeddings.most_similar(kg_word_embeddings.query(word), topn=1000) # Most similar by vector
            # print(top)
            candidates.extend([entity_score[0] for entity_score in top])
        # print candidates
        hits += len(set.intersection(set(candidates), set(existing_correct_entities)))
        # print hits
        i += 1
        # enough
        if i == n_samples:
            break

    print ("%d questions"%i)
    print ("%d entities"%n_entities)
    print ("%d entities in KG"%n_entities_kg)
    print ("%d entities in candidates"%hits)
    print ("%d missed entities"%(n_entities_kg-hits))
    # analyse the error
    # print(kg_word_embeddings.distance("director", "direct"))


if __name__ == '__main__':
    # produce_word_list()
    # Then generate fastText embeddings for both lists! fasttext + magnitude
    test_embeddings()
