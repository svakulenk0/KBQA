#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Aug 24, 2018

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Baseline neural network architecture for KBQA

Based on https://github.com/DSTC-MSR-NLP/DSTC7-End-to-End-Conversation-Modeling/blob/master/baseline/baseline.py 

Question - text as the sequence of words (word embeddings index)
Answer - entity vector from KB (entity embeddings index)


'''
import os
import wget
import zipfile
import json

import random
import numpy as np

import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

from keras.models import Model
from keras.models import load_model

from keras.layers import Input, GRU, Dropout, Embedding, Dense, Flatten
from keras.regularizers import l2
from keras.optimizers import Adam

from keras import backend as K

from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

from toy_data import *


EMBEDDINGS_PATH = "./embeddings/"
GLOVE_EMBEDDINGS_PATH = "./embeddings/glove.6B.50d.txt"
# rdf2vec embeddings 200 dimensions
KB_EMBEDDINGS_PATH = "/data/globalRecursive/data.dws.informatik.uni-mannheim.de/rdf2vec/models/DBpedia/2016-04/GlobalVectors/11_pageRankSplit/DBpediaVecotrs200_20Shuffle.txt"
# subset of the KB embeddings (rdf2vec embeddings 200 dimensions from KB_EMBEDDINGS_PATH) for the entities of the LC-Quad dataset (both train and test split)
LCQUAD_KB_EMBEDDINGS_PATH = "./data/selectedEmbeddings_lcquad_answers_train_1_test_all.txt"


def set_random_seed(seed=912):
    random.seed(seed)
    np.random.seed(seed)


# util creates missing folders
def makedirs(fld):
    if not os.path.exists(fld):
        os.makedirs(fld)


# Prepare Glove File
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
            wordToGlove[token] = np.array(record[1:], dtype=np.float64) # associate the Glove embedding vector to a that token (word)

        tokens = sorted(wordToGlove.keys())
        for idx, tok in enumerate(tokens):
            kerasIdx = idx + 1  # 0 is reserved for masking in Keras
            wordToIndex[tok] = kerasIdx # associate an index to a token (word)
            indexToWord[kerasIdx] = tok # associate a word to a token (word). Note: inverse of dictionary above

    return wordToIndex, indexToWord, wordToGlove

def create_KB_input(embeddings):
    '''
    Create an input which can be used in the network based on the existing embeddings
    '''
    from keras import backend as K
    k_constants = K.variable(embeddings)
    fixed_input = Input(tensor=k_constants)
    return fixed_input

def load_KB_embeddings(KB_embeddings_file=KB_EMBEDDINGS_PATH):
    '''
    load all embeddings from file
    '''
    entity2vec = {}

    print("Loading embeddings...")
    
    idx = 0
    entity2index = {}  # map from a token to an index
    index2entity = {}  # map from an index to a token 

    with open(KB_embeddings_file) as embs_file:
        # embeddings in a text file one per line for Global vectors and glove word embeddings
        for line in embs_file:
            entityAndVector = line.split(None, 1)
            # match the entity labels in vector embeddings
            entity = entityAndVector[0][1:-1]  # Dbpedia global vectors strip <> to match the entity labels
            try:
                embedding_vector = np.asarray(entityAndVector[1].split(), dtype='float32')
            except:
                print entityAndVector

            idx += 1  # 0 is reserved for masking in Keras
            entity2index[entity] = idx
            index2entity[idx] = entity
            entity2vec[entity] = embedding_vector
            n_dimensions = len(embedding_vector)

    print("Loaded %d embeddings with %d dimensions" % (len(entity2vec), n_dimensions))

    return (entity2index, index2entity, entity2vec, n_dimensions)


class KBQA:
    '''
    Baseline neural network architecture for KBQA
    '''
    def __init__(self, max_seq_len, rnn_units, encoder_depth, decoder_depth, num_hidden_units, bases, l2norm, dropout_rate=0.2, model_dir='./models/'):
        self.max_seq_len = max_seq_len
        self.rnn_units = rnn_units
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth
        self.num_hidden_units = num_hidden_units
        self.bases = bases
        self.l2norm = l2norm
        self.dropout_rate = dropout_rate
        makedirs(model_dir)
        self.model_dir = model_dir
        # load word vocabulary
        self.wordToIndex, self.indexToWord, self.wordToGlove = readGloveFile()

    def _stacked_rnn(self, rnns, inputs, initial_states=None):
        # if initial_states is None:
        #     initial_states = [None] * len(rnns)
        # outputs, state = rnns[0](inputs, initial_state=initial_states[0])
        outputs = rnns[0](inputs)
        # states = [state]
        for i in range(1, len(rnns)):
            # outputs, state = rnns[i](outputs, initial_state=initial_states[i])
            outputs = rnns[i](outputs)
            # states.append(state)
        return outputs

    def load_embeddings_from_index(self, embeddings_index, items_index):
        # load embeddings into matrix
        vocab_len = len(items_index) + 1  # adding 1 to account for masking
        embDim = next(iter(embeddings_index.values())).shape[0]
        embeddings_matrix = np.zeros((vocab_len, embDim))  # initialize with zeros
        for item, index in items_index.items():
            embeddings_matrix[index, :] = embeddings_index[item] # create embedding: item index to item embedding
        return embeddings_matrix

    def create_pretrained_embedding_layer(self, isTrainable=False):
        '''
        Create pre-trained Keras embedding layer
        '''
        self.word_vocab_len = len(self.wordToIndex) + 1  # adding 1 to account for masking
        embeddings_matrix = self.load_embeddings_from_index(self.wordToGlove, self.wordToIndex)

        embeddingLayer = Embedding(self.word_vocab_len, embeddings_matrix.shape[1], weights=[embeddings_matrix], trainable=isTrainable, name='word_embedding')
        return embeddingLayer

    def build_model_train(self):
        '''
        build layers required for training the NN
        '''
        # Q - question input
        question_input = Input(shape=(None,), name='question_input')

        # I - positive/negtive sample indicator (1/-1)
        # sample_input = Input(shape=(None,), name='sample_indicator')

        # E' - question words embedding
        word_embedding = self.create_pretrained_embedding_layer()
        
        # Q' - question encoder
        question_encoder_output_1 = GRU(self.rnn_units, name='question_encoder_1', return_sequences=True)(word_embedding(question_input))
        question_encoder_output_2 = GRU(self.rnn_units, name='question_encoder_2', return_sequences=True)(question_encoder_output_1)
        question_encoder_output = GRU(self.rnn_units, name='question_encoder_3')(question_encoder_output_2)

        print("%d samples of max length %d with %d hidden layer dimensions"%(self.num_samples, self.max_seq_len, self.rnn_units))
        
        answer_output = Dropout(self.dropout_rate)(question_encoder_output)

        self.model_train = Model(question_input,   # [input question, input KB],
                                 answer_output)                        # ground-truth target answer
        print self.model_train.summary()

    def load_data(self, dataset, split):
        questions, answers = dataset
        assert len(questions) == len(answers)

        # encode questions and answers using embeddings vocabulary
        num_samples = len(questions)
        self.entities = self.entity2vec.keys()

        questions_data = []
        answers_data = []
        answers_indices = []
        not_found_entities = 0

        # iterate over samples
        for i in range(num_samples):
            # encode words (ignore OOV words)
            questions_sequence = [self.wordToIndex[word] for word in text_to_word_sequence(questions[i]) if word in self.wordToIndex]
            answers_to_question = answers[i]
            
            if split == 'train':
                # train only on the first answer from the answer set
                first_answer = answers_to_question[0].encode('utf-8')
                # filter out answers without pre-trained embeddings
                if first_answer in self.entities:
                    # TODO match unicode lookup
                    questions_data.append(questions_sequence)
                    answers_data.append(self.entity2vec[first_answer])

                    # generate a random negative sample for each positive sample
                    questions_data.append(questions_sequence)
                    # pick a random entity
                    random_entity = random.choice(self.entities)
                    answers_data.append(self.entity2vec[random_entity])

            if split == 'test':
                # add all answer indices for testing
                answer_indices = []
                for answer in answers_to_question:
                    answer = answer.encode('utf-8')
                    if answer in self.entity2vec.keys():
                        answer_indices.append(self.entity2index[answer])

                answers_indices.append(answer_indices)
                # if answer_indices:
                questions_data.append(questions_sequence)

            
            # else:
            #     not_found_entities +=1
        
        print ("Not found: %d entities"%not_found_entities)
        # normalize length
        questions_data = np.asarray(pad_sequences(questions_data, padding='post'))
        print("Maximum question length %d"%questions_data.shape[1])
        answers_data = np.asarray(answers_data)

        self.num_samples = questions_data.shape[0]
       
        # print questions_data
        # print answers_data

        self.dataset = (questions_data, answers_data, answers_indices)
        print("Loaded the dataset")

    def save_model(self, name):
        path = os.path.join(self.model_dir, name)
        self.model_train.save(path)
        print('Saved to: '+path)

    def load_pretrained_model(self):
        self.model_train = load_model(os.path.join(self.model_dir, 'model.h5'))
        # self.build_model_test()

    def samples_loss(self):
        def loss(y_true, y_pred):
            y_true = K.l2_normalize(y_true, axis=-1)
            y_pred = K.l2_normalize(y_pred, axis=-1)
            # print("Batch size: %s" % str(y_pred.shape))
            size = K.variable(K.shape(y_pred)[0]/2)
            indicator = K.variable(value=[1, -1])
            loss_vector = -K.sum(y_true * y_pred, axis=-1) * K.tile(indicator, size)
            print("Loss: %s" % str(loss_vector.shape))
            return loss_vector
        return loss

    def train(self, batch_size, epochs, batch_per_load=10, lr=0.001):
        questions_vectors, answers_vectors, answers_indices = self.dataset
        self.model_train.compile(optimizer=Adam(lr=lr), loss=self.samples_loss())
        # for epoch in range(epochs):
        #     print('\n***** Epoch %i/%i *****'%(epoch + 1, epochs))
            # load training dataset
            # encoder_input_data, decoder_input_data, decoder_target_data, _, _ = self.dataset.load_data('train', batch_size * batch_per_load)
            # self.model_train.fit([questions] +[X] + A, answers, batch_size=batch_size,)
        self.model_train.fit(questions_vectors, answers_vectors, epochs=epochs, verbose=2, validation_split=0.3, shuffle='batch')
        self.save_model('model.h5')
            # self.save_model('model_epoch%i.h5'%(epoch + 1))
        # self.save_model('model.h5')

    def test(self):
        questions_vectors, answers_vectors, answers_indices = self.dataset
        print("Testing...")
        # score = self.model_train.evaluate(questions, answers, verbose=0)
        # print score
        print("Questions vectors shape: " + " ".join([str(dim) for dim in questions_vectors.shape]))
        # print("Answers vectors shape: " + " ".join([str(dim) for dim in answers_vectors.shape]))
        print("Answers indices shape: %d" % len(answers_indices))

        predicted_answers_vectors = self.model_train.predict(questions_vectors)
        print("Predicted answers vectors shape: " + " ".join([str(dim) for dim in predicted_answers_vectors.shape]))
        # print("Answers indices: " + ", ".join([str(idx) for idx in answers_indices]))

        # load embeddings into matrix
        embeddings_matrix = self.load_embeddings_from_index(self.entity2vec, self.entity2index)
        # calculate pairwise distances (via cosine similarity)
        similarity_matrix = cosine_similarity(predicted_answers_vectors, embeddings_matrix)

        # print np.argmax(similarity_matrix, axis=1)

        n = 5
        # indices of the top n predicted answers for every question in the test set
        top_ns = similarity_matrix.argsort(axis=1)[:, -n:][::-1]
        # print top_ns[:2]

        hits = 0
        for i, answers in enumerate(answers_indices):
            # check if the correct and predicted answer sets intersect
            if set.intersection(set(answers), set(top_ns[i])):
            # if set.intersection(set([answers[0]]), set(top_ns[i])):
                hits += 1

        print("Hits in top %d: %d/%d"%(n, hits, len(answers_indices)))


def download_glove_embeddings():
    makedirs(EMBEDDINGS_PATH)

    if not os.path.exists(GLOVE_EMBEDDINGS_PATH):
        wget.download('http://nlp.stanford.edu/data/glove.6B.zip', EMBEDDINGS_PATH+"glove.6B.zip")
        with zipfile.ZipFile(EMBEDDINGS_PATH+"glove.6B.zip","r") as zip_ref:
            zip_ref.extractall(EMBEDDINGS_PATH)


def load_lcquad(dataset_split):
    # load embeddings
    entity2index, index2entity, entity2vec, kb_embeddings_dimension = load_KB_embeddings(LCQUAD_KB_EMBEDDINGS_PATH)
    QS = []
    AS = []
    with open("./data/lcquad_%s.json"%dataset_split, "r") as train_file:
        qas = json.load(train_file)
        for qa in qas:
            QS.append(qa['question'])
            AS.append(qa['answers'])
    return (QS, AS), entity2index, index2entity, entity2vec, kb_embeddings_dimension


# def load_dbnqa():
#     return (QS, AS)


def load_toy_data():
    return (QS, AS), ENTITY2VEC, KB_EMBEDDINGS_DIM


def train_model(model):
    '''
    dataset_name <String> Choose one of the available datasets to train the model on ('toy', 'lcquad')
    '''
    # build model
    model.build_model_train()
    # train model
    model.train(batch_size, epochs, lr=learning_rate)


def test_model(model):
    '''
    dataset_name <String> Choose one of the available datasets to test the model on ('lcquad')
    '''
    model.load_pretrained_model()
    print("Loaded the pre-trained model")
    model.test()


def load_data(model, dataset_name, mode):
    print("Loading %s..."%dataset_name)
    
    if dataset_name == 'toy':
        dataset, model.entity2vec, model.kb_embeddings_dimension = load_toy_data()
    # elif dataset_name == 'dbnqa':
    #     dataset = load_dbnqa()
    elif dataset_name == 'lcquad':
        dataset, model.entity2index, model.index2entity, model.entity2vec, model.kb_embeddings_dimension = load_lcquad(mode)

    model.load_data(dataset, mode)


if __name__ == '__main__':
    set_random_seed()
    # set mode and dataset
    mode = 'train'
    dataset_name = 'lcquad'
    # dataset_name = 'lcquad_test'

    # define QA model architecture parameters
    max_seq_len = 10
    rnn_units = 200  # dimension of the GRU output layer (hidden question representation) 
    encoder_depth = 2
    decoder_depth = 2
    dropout_rate = 0.5

    # define R-GCN architecture parameters
    num_hidden_units = 16
    bases = -1
    l2norm = 0.

    # define training parameters
    batch_size = 100
    epochs = 20  # 10
    learning_rate = 1e-3

    # initialize the model
    model = KBQA(max_seq_len, rnn_units, encoder_depth, decoder_depth, num_hidden_units, bases, l2norm, dropout_rate)

    # load data
    load_data(model, dataset_name, mode)
    
    # modes
    if mode == 'train':
        train_model(model)
    elif mode == 'test':
        test_model(model)
