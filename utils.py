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

import scipy.sparse as sp

from keras import backend as K

# word embeddings
EMBEDDINGS_PATH = "./embeddings/"
GLOVE_EMBEDDINGS_PATH = "./embeddings/glove.6B.50d.txt"
FASTTEXT_MODEL_PATH = "/data/fasttext/wiki.en.bin"

# KG
DBPEDIA = './data/graph/data/dbpedia2016_04_run2/'
KG = DBPEDIA
ADJACENCY_MATRIX = KG + "adjacency.pickle"
ENTITIES_LIST = KG + "nodes_strings.pkl"

# KG embeddings
# rdf2vec embeddings 200 dimensions
RDF2VEC_EMBEDDINGS_PATH = "/data/globalRecursive/data.dws.informatik.uni-mannheim.de/rdf2vec/models/DBpedia/2016-04/GlobalVectors/11_pageRankSplit/DBpediaVecotrs200_20Shuffle.txt"
# subset of the KB embeddings (rdf2vec embeddings 200 dimensions from KB_EMBEDDINGS_PATH) for the entities of the LC-Quad dataset (both train and test split)
# LCQUAD_KB_EMBEDDINGS_PATH = "./data/selectedEmbeddings_lcquad_answers_train_1_test_all.txt"
# LCQUAD_KB_EMBEDDINGS_PATH = "./data/selectedEmbeddings_KGlove_PR_Split_lcquad_answers_train_1_test_all.txt"
# LCQUAD_KB_EMBEDDINGS_PATH = "./data/selectedEmbeddings_rdf2vec_uniform_lcquad_answers_train_1_test_all.txt"
KB_EMBEDDINGS_PATH = RDF2VEC_EMBEDDINGS_PATH


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
        dataset = pkl.load(f)
        kb_adjacency = dataset['A']
       
        if relation_limit:
            kb_adjacency = kb_adjacency[:relation_limit]
        
        # convert dense numpy adjacency matrices to sparse
        adjacencies = []
        for relation in kb_adjacency:

            # print ("Adjacency shape:", relation.shape)
            # adj_shape = (relation.shape, relation.shape)

            # split subject (row) and object (col) node URIs
            # row, col = np.transpose(relation)

            # create adjacency matrix for this property
            # data = np.ones(len(row), dtype=np.int8)
            adj = sp.csr_matrix(relation, dtype=np.int8)
            # if normalize:
            #     adj = normalize_adjacency_matrix(adj)
            adjacencies.append(adj)

            # create adjacency matrix for inverse property
            # if include_inverse:
            #     adj = sp.csr_matrix((data, (col, row)), shape=adj_shape, dtype=np.int8)
            #     # if normalize:
            #     #     adj = normalize_adjacency_matrix(adj)
            #     adjacencies.append(adj)

        # save reduced A
        # with open(KB+"adjacency_short.pickle", 'wb') as f:
        #     pkl.dump(kb_adjacency, f, pkl.HIGHEST_PROTOCOL)

    return entityToIndex, sp.hstack(adjacencies, format="csr")


def load_KB_embeddings(KB_embeddings_file=KB_EMBEDDINGS_PATH):
    '''
    load all embeddings from file
    '''
    entity2vec = {}
    entity2index = {}  # map from a token to an index
    index2entity = {}  # map from an index to a token 
    kg_embeddings_matrix = []

    with open(KB_embeddings_file) as embs_file:
        # embeddings in a text file one per line for Global vectors and glove word embeddings
        for idx, line in enumerate(embs_file):
            entityAndVector = line.split(None, 1)
            # match the entity labels in vector embeddings
            entity = entityAndVector[0][1:-1]  # Dbpedia global vectors strip <> to match the entity labels
            embedding_vector = np.asarray(entityAndVector[1].split(), dtype='float32')
            kg_embeddings_matrix.append(embedding_vector)
            entity2vec[idx] = embedding_vector

            entity2index[entity] = idx
            index2entity[idx] = entity

    kg_embeddings_matrix = np.asarray(kg_embeddings_matrix, dtype=K.floatx())

    return (entity2index, index2entity, entity2vec, kg_embeddings_matrix)


def load_fasttext(model_path=FASTTEXT_MODEL_PATH):
    import fastText
    return fastText.load_model(model_path)


def embed_with_fasttext(words):
    '''
    words <list> of words to be translated into vectors
    model_path <str> path to the pre-trained FastText model binary file
    '''
    model = load_fasttext()
    vectors = []
    for word in words:
        vectors.append(model.get_word_vector(word)) # produce a vector for the string
    return vectors


def my_loss():
    def loss(y_true, y_pred):
        print ("Predicted vectors: %s" % str(y_pred.shape))
        y_true = K.l2_normalize(y_true, axis=-1)
        y_pred = K.l2_normalize(y_pred, axis=-1)
        loss_vector = -K.sum(y_true * y_pred, axis=-1)
        return loss_vector
    return loss
