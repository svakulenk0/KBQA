#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Nov 4, 2018

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Evaluate question entity selection using word embeddings
'''
import io

from keras.preprocessing.text import text_to_word_sequence
from pymagnitude import *


test_q = "How many movies did Stanley Kubrick direct?"
kg_entities_path = './data/lcquad_train_entities.txt'
correct_entities = ["http://dbpedia.org/ontology/director", "http://dbpedia.org/resource/Stanley_Kubrick"]

def save_words_list(words_list, file_name):
    with open(file_name, 'w') as out:
        out.write('\n'.join(words_list) + '\n')


def produce_word_lists(questions=[test_q], kg_entities_path=kg_entities_path):
    # save a list of question words
    question_words = []
    for question in questions:
        question_words.extend([word for word in text_to_word_sequence(question)])
        assert len(question_words) == 7
    save_words_list(question_words, './data/test_question_words.txt')

    # save a list of entity labels
    entity_labels = []
    with open(kg_entities_path) as entities_file:
        for entity_uri in entities_file:
            # strip the domain name from the entity_uri to produce a cleaner entity label
            entity_label = entity_uri.strip('\n').strip('/').strip('>').split('/')[-1]
            entity_labels.append(entity_label)
    save_words_list(entity_labels, './data/lcquad_train_entities_labels.txt')


def test_embeddings(questions=[test_q], correct_entities=[correct_entities],
                    fname_kg='./data/lcquad_train_entities_labels_fasttext.magnitude'):
    
    # load embeddings
    kg_word_embeddings = Magnitude(fname_kg)

    # preprocess entities
    correct_entities = [entity_uri.strip('\n').strip('/').strip('>').split('/')[-1] for entity_uri in correct_entities]
    print correct_entities
    
    # iterate over questions
    i = 0
    for question in questions:
        
        candidates = []
        
        for word in text_to_word_sequence(question):
            print(word)
            # top_key = kg_word_embeddings.most_similar(word, topn=100) # Most similar by key
            # print(top)
            top = kg_word_embeddings.most_similar(kg_word_embeddings.query(word), topn=10) # Most similar by vector
            candidates.extend(top)

        hits = set.intersection(set(candidates), set(correct_entities[i]))
        print hits
        i += 1


if __name__ == '__main__':
    # produce_word_lists()
    # Then generate fastText embeddings for both lists!
    test_embeddings()
