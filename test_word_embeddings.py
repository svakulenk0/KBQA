#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Nov 4, 2018

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Evaluate question entity selection using word embeddings
'''
from keras.preprocessing.text import text_to_word_sequence
from sklearn.metrics.pairwise import cosine_similarity


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

# generate fastText embeddings for both lists

# TODO
# load all entities from file and embed
    # kg_word_embeddings = []
    # # store entity vocabulary
    # ent2idx = {}
    # idx2ent = {}
    # with open(kg_entities_path) as entities_file:
    #     idx = 0
    #     for entity_uri in entities_file:
    #         ent2idx[entity_uri] = idx
    #         idx2ent[idx] = entity_uri
    #         # strip the domain name from the entity_uri to produce a cleaner entity label
    #         entity_label = entity_uri.strip('/').split('/')[-1]
    #         # embed the produced entity label into the same word vector space
    #         kg_word_embeddings.append(wordToVec.get_word_vector(entity_label))

    # # compute text similarity (cosine) between the question words and KG entity labels: all question words x all entities
    # similarity_matrix = cosine_similarity(question_word_embeddings, kg_word_embeddings)

    # # indices of the top n similar entities for every question word
    # top_n = 5
    # top_ns = similarity_matrix.argsort(axis=1)[:, -n:][::-1]
    # print [idx2ent[idx] for word_top_ns in top_ns for idx in word_top_ns]

# check the rank of the correct_entities
# 

if __name__ == '__main__':
    produce_word_lists()
