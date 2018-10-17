#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Aug 30, 2018

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Utils functions for different KBQA models
'''
import os
import json
import pickle as pkl

import numpy as np
import random

from keras import backend as K

# word embeddings
EMBEDDINGS_PATH = "./embeddings/"
GLOVE_EMBEDDINGS_PATH = "./embeddings/glove.6B.50d.txt"

# KB
DBPEDIA = './data/graph/data/dbpedia2016_04_run2/'

KB = DBPEDIA
# ADJACENCY_MATRIX = KB + "adjacency.pickle"
ADJACENCY_MATRIX = KB + "adjacency_short.pickle"
ENTITIES_LIST = KB + "nodes_strings.pkl"


def set_random_seed(seed=912):
    random.seed(seed)
    np.random.seed(seed)


def load_dbnqa():
    pass


def load_lcquad(dataset_split):
    QS = []
    AS = []
    with open("./data/lcquad_%s.json"%dataset_split, "r") as train_file:
        qas = json.load(train_file)
        for qa in qas:
            QS.append(qa['question'])
            AS.append(qa['answers'])
    return (QS, AS)


def load_dataset(dataset_name, dataset_split):
    print("Loading %s..."%dataset_name)
    
    
    if dataset_name == 'lcquad':
        dataset = load_lcquad(dataset_split)
    elif dataset_name == 'dbnqa':
        dataset = load_dbnqa()

    return dataset


# util creates missing folders
def makedirs(fld):
    if not os.path.exists(fld):
        os.makedirs(fld)


def download_glove_embeddings():
    makedirs(EMBEDDINGS_PATH)

    if not os.path.exists(GLOVE_EMBEDDINGS_PATH):
        wget.download('http://nlp.stanford.edu/data/glove.6B.zip', EMBEDDINGS_PATH+"glove.6B.zip")
        with zipfile.ZipFile(EMBEDDINGS_PATH+"glove.6B.zip","r") as zip_ref:
            zip_ref.extractall(EMBEDDINGS_PATH)


def readGloveFile(gloveFile=GLOVE_EMBEDDINGS_PATH):
    '''
    https://stackoverflow.com/questions/48677077/how-do-i-create-a-keras-embedding-layer-from-a-pre-trained-word-embedding-datase
    '''
    download_glove_embeddings()

    with open(gloveFile, 'r') as f:
        wordToGlove = {}  # map from a token (word) to a Glove embedding vector
        wordToIndex = {}  # map from a token to an index
        indexToWord = {}  # map from an index to a token 

        for line in f:
            record = line.strip().split()
            token = record[0] # take the token (word) from the text line
            wordToGlove[token] = np.array(record[1:], dtype=K.floatx()) # associate the Glove embedding vector to a that token (word)

        tokens = sorted(wordToGlove.keys())
        for idx, tok in enumerate(tokens):
            kerasIdx = idx + 1  # 0 is reserved for masking in Keras
            wordToIndex[tok] = kerasIdx # associate an index to a token (word)
            indexToWord[kerasIdx] = tok # associate a word to a token (word). Note: inverse of dictionary above

    return wordToIndex, indexToWord, wordToGlove


def loadKB(kb_entity_labels_list=ENTITIES_LIST, kb_adjacency_path=ADJACENCY_MATRIX, entity_limit=None, relation_limit=None):
    '''
    Returns an index of entities <dict> and adjacency matrix
    <str> "entity_label": <int> index
    where 0 is a mask symbol in Keras
    e.g. {'http://dbpedia.org/resource/Pittsburgh': 1}
    '''

    # index entity labels into a map
    entityToIndex = {}
    with open(kb_entity_labels_list, 'rb') as f:
        entity_labels = pkl.load(f)
        for idx, entity_label in enumerate(entity_labels):
            keras_idx = idx + 1  # mask 0 for padding
            entityToIndex[entity_label] = keras_idx
            if entity_limit:
                if keras_idx >= entity_limit:
                    print("%d limit on the number of entities"%entity_limit)
                    break
    # generate adjacency matrix for each property
    with open(kb_adjacency_path, 'rb') as f:
        # data = pkl.load(f, encoding='ISO-8859-1')
        kb_adjacency = pkl.load(f)
        if relation_limit:
            kb_adjacency = kb_adjacency[:relation_limit]
        print ("Adjacency shape:", kb_adjacency.shape)
        for relation in kb_adjacency:
            if entity_limit:
                relation = relation[:entity_limit]
                adj_shape = (entity_limit, entity_limit)
            # split subject (row) and object (col) node URIs
            row, col = np.transpose(relation)

            # create adjacency matrix for this property
            data = np.ones(len(row), dtype=np.int8)
            adj = sp.csr_matrix((data, (row, col)), shape=adj_shape, dtype=np.int8)
            # if normalize:
            #     adj = normalize_adjacency_matrix(adj)
            adjacencies.append(adj)

            # create adjacency matrix for inverse property
            if include_inverse:
                adj = sp.csr_matrix((data, (col, row)), shape=adj_shape, dtype=np.int8)
                # if normalize:
                #     adj = normalize_adjacency_matrix(adj)
                adjacencies.append(adj)

        # save reduced A
        # with open(KB+"adjacency_short.pickle", 'wb') as f:
        #     pkl.dump(kb_adjacency, f, pkl.HIGHEST_PROTOCOL)

    return entityToIndex, adjacencies


def load_embeddings_from_index(embeddings_index, items_index):
    # load embeddings into matrix
    vocab_len = len(items_index) + 1  # adding 1 to account for masking
    embDim = next(iter(embeddings_index.values())).shape[0]
    embeddings_matrix = np.zeros((vocab_len, embDim))  # initialize with zeros
    for item, index in items_index.items():
        embeddings_matrix[index, :] = embeddings_index[item] # create embedding: item index to item embedding
    return embeddings_matrix
