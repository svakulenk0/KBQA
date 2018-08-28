#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Aug 24, 2018

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Translation-based neural network model for KBQA

Based on https://github.com/DSTC-MSR-NLP/DSTC7-End-to-End-Conversation-Modeling/blob/master/baseline/baseline.py 

Question - text as the sequence of words (word embeddings index)
Answer - entity vector from KB (entity embeddings index)


'''
import os
import wget
import zipfile
import json

import numpy as np
import scipy.sparse as sp

from keras.models import Model
from keras.layers import Input, GRU, Dropout, Embedding, Dense, Flatten
from keras.regularizers import l2
from keras.optimizers import Adam

from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

from toy_data import *


EMBEDDINGS_PATH = "./embeddings/"
GLOVE_EMBEDDINGS_PATH = "./embeddings/glove.6B.50d.txt"
# rdf2vec embeddings 200 dimensions
KB_EMBEDDINGS_PATH = "/data/globalRecursive/data.dws.informatik.uni-mannheim.de/rdf2vec/models/DBpedia/2016-04/GlobalVectors/11_pageRankSplit/DBpediaVecotrs200_20Shuffle.txt"
# LCQUAD_KB_EMBEDDINGS_PATH = "./data/selectedEmbeddings_lcquad_train_answers.txt"
LCQUAD_KB_EMBEDDINGS_PATH = "./data/selectedEmbeddings_lcquad_train_answer_5000_uniq.txt"


# Prepare Glove File
def readGloveFile(gloveFile=GLOVE_EMBEDDINGS_PATH):
    '''
    https://stackoverflow.com/questions/48677077/how-do-i-create-a-keras-embedding-layer-from-a-pre-trained-word-embedding-datase
    '''
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

    with open(KB_embeddings_file) as embs_file:
        # embeddings in a text file one per line for Global vectors and glove word embeddings
        for line in embs_file:
            entityAndVector = line.split(None, 1)
            # match the entity labels in vector embeddings
            entity = entityAndVector[0][1:-1]  # Dbpedia global vectors strip <> to match the entity labels
            embedding_vector = np.asarray(entityAndVector[1].split(), dtype='float32')
            n_dimensions = len(embedding_vector)
            entity2vec[entity] = embedding_vector

    print("Loaded %d embeddings with %d dimensions" % (len(entity2vec), n_dimensions))

    return (entity2vec, n_dimensions)


class KBQA_Translation:
    '''
    Translation-based NN model for KBQA
    '''
    def __init__(self, max_seq_len, rnn_units, encoder_depth, decoder_depth, num_hidden_units, bases, l2norm, dropout_rate=0.2):
        self.max_seq_len = max_seq_len
        self.rnn_units = rnn_units
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth
        self.num_hidden_units = num_hidden_units
        self.bases = bases
        self.l2norm = l2norm
        self.dropout_rate = dropout_rate

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

    def create_pretrained_embedding_layer(self, isTrainable=False):
        '''
        Create pre-trained Keras embedding layer
        '''
        self.word_vocab_len = len(self.wordToIndex) + 1  # adding 1 to account for masking
        embDim = next(iter(self.wordToGlove.values())).shape[0]  # works with any glove dimensions (e.g. 50)

        embeddingMatrix = np.zeros((self.word_vocab_len, embDim))  # initialize with zeros
        for word, index in self.wordToIndex.items():
            embeddingMatrix[index, :] = self.wordToGlove[word] # create embedding: word index to Glove word embedding

        embeddingLayer = Embedding(self.word_vocab_len, embDim, weights=[embeddingMatrix], trainable=isTrainable, name='word_embedding')
        return embeddingLayer

    def build_model_train(self, dataset):
        '''
        build layers
        '''
        # Q' - question encoder
        question_input = Input(shape=(None,), name='question_input')

        # E' - question words embedding
        self.wordToIndex, self.indexToWord, self.wordToGlove = readGloveFile()
        word_embedding = self.create_pretrained_embedding_layer()

        # E'' - answer entities (KB) embedding

        # self.wordToIndex, self.indexToWord, self.wordToGlove = readGloveFile()
        # word_embedding = self.create_pretrained_embedding_layer()

        # load data
        self.load_data(dataset)

        question_encoder_output = GRU(self.rnn_units, name='question_encoder')(word_embedding(question_input))
        # question_encoder = []
        # for i in range(self.encoder_depth):
        #     question_encoder.append(GRU(
        #         self.rnn_units, 
        #         # return_state=True,
        #         # return_sequences=True, 
        #         name='question_encoder_%i' % i
        #         ))

        # K' - KB encoder layer: (entities, relations) adjacency matrix as input via R-GCN architecture
        # https://github.com/tkipf/relational-gcn/blob/master/rgcn/train.py

            # A_in = [InputAdj(sparse=True) for _ in range(self.support)]
            # X_in = Input(shape=(self.entity_vocab_len,), sparse=True)

            # kb_encoder_input = [X_in] + A_in
            # input=[X_in] + A_in

            # # E'' - KB entities initial embedding
            # # entity_embedding = 

            # kb_encoder = GraphConvolution(self.num_hidden_units, self.support, num_bases=self.bases, featureless=True,
            #                               activation='relu',
            #                               W_regularizer=l2(self.l2norm))
        # kb_encoder_output = GraphConvolution(self.num_hidden_units, support, num_bases=self.bases, featureless=True,
                             # activation='relu',
                             # W_regularizer=l2(self.l2norm))(kb_encoder_input)
        # kb_encoder = Dropout(self.dropout_rate)(kb_encoder)
        # kb_encoder = GraphConvolution(, support, num_bases=self.bases,
                             # activation='softmax')([kb_encoder] + A_in)

        
        # A' - answer decoder
        # answer_decoder = []
        # for i in range(self.decoder_depth):
        #     answer_decoder.append(GRU(
        #         self.rnn_units, 
        #         # return_state=True,
        #         return_sequences=True, 
        #         name='answer_decoder_%i'%i,
        #         ))

        # decoder_softmax = Dense(self.entity_vocab_len, activation='softmax', name='decoder_softmax')

        # network architecture
        # question_encoder_output = self._stacked_rnn(question_encoder, word_embedding(question_input))
            # kb_encoder_output = kb_encoder(kb_encoder_input)
        
        # to do join outputs of the encoders and decoder

        # decoder_outputs, decoder_states = self._stacked_rnn(answer_decoder, question_encoder_output + kb_encoder_output, [question_encoder_states[-1]] * self.decoder_depth)
        # decoder_outputs = self._stacked_rnn(answer_decoder, question_encoder_output + kb_encoder_output)
        # question_encoder_output = Dropout(self.dropout_rate)(question_encoder_output)
        # kb_encoder_output = Dropout(self.dropout_rate)(kb_encoder_output)
        # decoder_outputs = decoder_softmax(decoder_outputs)

        # fix: add output of the KB encoder
        # answer_decoder_output = decoder_softmax(question_encoder_output)

        # reshape question_encoder_output to the answer embedding vector size
        # answer_output = Reshape((self.kb_embeddings_dimension,), input_shape=(self.max_seq_len, self.rnn_units))(question_encoder_output)
        print (self.num_samples, self.max_seq_len, self.rnn_units)
        # answer_output = Flatten(input_shape=(self.num_samples, self.max_seq_len, self.rnn_units))(question_encoder_output)
        answer_output = question_encoder_output

        # self.model_train = Model([question_encoder_input] +[X_in] + A_in,   # [input question, input KB],
        self.model_train = Model(question_input,   # [input question, input KB],
                                 answer_output)                        # ground-truth target answer
        print self.model_train.summary()

    def load_data(self, dataset):
        questions, answers = dataset
        assert len(questions) == len(answers)

        # encode entities with one-hot-vector encoding
            # X = sp.csr_matrix(A[0].shape)
        # self.entityToIndex = {}

        # # todo generate entity index
        # self.entityToIndex = {}
        # self.entity_vocab_len = self.word_vocab_len ## X.shape[1]

        # define KB parameters for input to R-GCN 
        # self.support = len(A)
        # self.num_entities = X.shape[1]

        # encode questions and answers using embeddings vocabulary
        self.num_samples = len(questions)
        # num_samples = 1

        # questions_data = np.zeros((self.num_samples, self.max_seq_len))
        # answers_data = np.zeros((num_samples, self.max_seq_len, self.entity_vocab_len))
        questions_data = []
        answers_data = []
        not_found_entities = 0
        
        # iterate over samples
        for i in range(self.num_samples):
            # encode words (ignore OOV words)
            questions_sequence = [self.wordToIndex[word] for word in text_to_word_sequence(questions[i]) if word in self.wordToIndex]
            # for t, token_index in enumerate(questions_sequence):
                # questions_data[i, t] = token_index
            # print len(self.entity2vec[answers[i]])
            answer = answers[i]

            # filter out answers without pre-trained embeddings
            if answer in self.entity2vec.keys():
                questions_data.append(questions_sequence)
                # TODO match unicode lookup
                answers_data.append(self.entity2vec[answer])
            else:
                not_found_entities +=1
            # encode answer into a one-hot-encoding with a 3 dimensional tensor
            # answers_sequence = [self.wordToIndex[word] for word in text_to_word_sequence(answers[0]) if word in self.wordToIndex]
            # for t, token_index in enumerate(answers_sequence):
            #     answers_data[i, t, token_index] = 1.
        
        print ("Not found: %d entities"%not_found_entities)
        # normalize length
        questions_data = np.asarray(pad_sequences(questions_data, padding='post'))
        print(questions_data.shape[1])
        answers_data = np.asarray(answers_data)
       
        # print questions_data
        # print answers_data

        self.dataset = (questions_data, answers_data)

    def train(self, batch_size, epochs, batch_per_load=10, lr=0.001):
        self.model_train.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy')
        
        questions, answers = self.dataset

        # for epoch in range(epochs):
        #     print('\n***** Epoch %i/%i *****'%(epoch + 1, epochs))
            # load training dataset
            # encoder_input_data, decoder_input_data, decoder_target_data, _, _ = self.dataset.load_data('train', batch_size * batch_per_load)
            # self.model_train.fit([questions] +[X] + A, answers, batch_size=batch_size,)
        self.model_train.fit(questions, answers, epochs=epochs, verbose=2)

            # self.save_model('model_epoch%i.h5'%(epoch + 1))
        # self.save_model('model.h5')


def download_glove_embeddings():
    if not os.path.exists(EMBEDDINGS_PATH):
        os.makedirs(EMBEDDINGS_PATH)

    if not os.path.exists(GLOVE_EMBEDDINGS_PATH):
        wget.download('http://nlp.stanford.edu/data/glove.6B.zip', EMBEDDINGS_PATH+"glove.6B.zip")
        with zipfile.ZipFile(EMBEDDINGS_PATH+"glove.6B.zip","r") as zip_ref:
            zip_ref.extractall(EMBEDDINGS_PATH)


def load_lcquad():
    # load embeddings
    entity2vec, kb_embeddings_dimension = load_KB_embeddings(LCQUAD_KB_EMBEDDINGS_PATH)
    QS = []
    AS = []
    with open("./data/lcquad_train.json", "r") as train_file:
        qas = json.load(train_file)
        for qa in qas:
            QS.append(qa['question'])
            AS.append(qa['answers'][0])
    return (QS, AS), entity2vec, kb_embeddings_dimension


# def load_dbnqa():
#     return (QS, AS)


def load_toy_data():
    return (QS, AS), ENTITY2VEC, KB_EMBEDDINGS_DIM


def train_model(dataset_name):
    '''
    dataset_name <String> Choose one of the available datasets to train the model on ('toy', 'dbnqa')
    '''
    download_glove_embeddings()

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
    epochs = 50  # 10
    learning_rate = 1e-3

    # initialize the model
    model = KBQA_Translation(max_seq_len, rnn_units, encoder_depth, decoder_depth, num_hidden_units, bases, l2norm, dropout_rate)

    if dataset_name == 'toy':
        dataset, model.entity2vec, model.kb_embeddings_dimension = load_toy_data()
    # elif dataset_name == 'dbnqa':
    #     dataset = load_dbnqa()
    elif dataset_name == 'lcquad':
        dataset, model.entity2vec, model.kb_embeddings_dimension = load_lcquad()


    # build model
    model.build_model_train(dataset)
   
    # train model
    model.train(batch_size, epochs, lr=learning_rate)


if __name__ == '__main__':
    # load_KB_embeddings()
    dataset_name = 'lcquad'
    train_model(dataset_name)

    # set mode
    # mode = 'train'
    
    # # modes
    # if mode == 'train':
