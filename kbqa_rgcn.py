#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Aug 22, 2018

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Neural network model with R-GCN layer for KBQA

Based on https://github.com/DSTC-MSR-NLP/DSTC7-End-to-End-Conversation-Modeling/blob/master/baseline/baseline.py 

'''
import os
import wget
import zipfile

import numpy as np

from keras.layers import Input, GRU, Dropout, Embedding, Dense
from keras.regularizers import l2

from rgcn.layers.graph import GraphConvolution
from rgcn.layers.input_adj import InputAdj

EMBEDDINGS_PATH = "./embeddings/"
GLOVE_EMBEDDINGS_PATH = "./embeddings/glove.6B.50d.txt"


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
            kerasIdx = idx + 1  # 0 is reserved for masking in Keras (see above)
            wordToIndex[tok] = kerasIdx # associate an index to a token (word)
            indexToWord[kerasIdx] = tok # associate a word to a token (word). Note: inverse of dictionary above

    return wordToIndex, indexToWord, wordToGlove


def create_pretrained_embedding_layer(wordToGlove, wordToIndex, isTrainable):
    '''
    Create pre-trained Keras embedding layer
    '''
    vocabLen = len(wordToIndex) + 1  # adding 1 to account for masking
    embDim = next(iter(wordToGlove.values())).shape[0]  # works with any glove dimensions (e.g. 50)

    embeddingMatrix = np.zeros((vocabLen, embDim))  # initialize with zeros
    for word, index in wordToIndex.items():
        embeddingMatrix[index, :] = wordToGlove[word] # create embedding: word index to Glove word embedding

    embeddingLayer = Embedding(vocabLen, embDim, weights=[embeddingMatrix], trainable=isTrainable, name='word_embedding')
    return embeddingLayer


class KBQA_RGCN:
    '''
    NN model for KBQA with R-GCN for KB embedding training
    '''
    def __init__(self, rnn_units, encoder_depth, decoder_depth, num_hidden_units, bases, l2norm, dropout_rate=0.2):
        self.rnn_units = rnn_units
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth
        self.num_hidden_units = num_hidden_units
        self.bases = bases
        self.l2norm = l2norm
        self.dropout_rate = dropout_rate

    def _stacked_rnn(self, rnns, inputs, initial_states=None):
        if initial_states is None:
            initial_states = [None] * len(rnns)
        outputs, state = rnns[0](inputs, initial_state=initial_states[0])
        states = [state]
        for i in range(1, len(rnns)):
            outputs, state = rnns[i](outputs, initial_state=initial_states[i])
            states.append(state)
        return outputs, states

    def build_model_train(self):
        '''
        build layers
        '''
        # Q' - question encoder
        question_encoder_input = Input(shape=(None,), name='question_encoder_input')

        # E' - question words embedding
        wordToIndex, indexToWord, wordToGlove = readGloveFile()
        word_embedding = create_pretrained_embedding_layer(wordToGlove, wordToIndex, False)

        question_encoder = []
        for i in range(self.encoder_depth):
            question_encoder.append(GRU(
                self.rnn_units, 
                # return_state=True,
                return_sequences=True, 
                name='question_encoder_%i' % i
                ))

        # K' - KB encoder layer: (entities, relations) adjacency matrix as input via R-GCN architecture
        # https://github.com/tkipf/relational-gcn/blob/master/rgcn/train.py
        
        # define KB parameters for input to R-GCN 
        # support = len(A)
        support = 100
        # num_entities = X.shape[1]
        num_entities = 1000

        A_in = [InputAdj(sparse=True) for _ in range(support)]
        X_in = Input(shape=(num_entities,), sparse=True)

        kb_encoder_input = [X_in] + A_in

        # E'' - KB entities initial embedding
        # entity_embedding = 

        # kb_encoder
        # kb_encoder_output = kb_encoder(kb_encoder_input)
        kb_encoder_output = GraphConvolution(self.num_hidden_units, support, num_bases=self.bases, featureless=True,
                             activation='relu',
                             W_regularizer=l2(self.l2norm))(kb_encoder_input)
        # kb_encoder = Dropout(self.dropout_rate)(kb_encoder)
        # kb_encoder = GraphConvolution(, support, num_bases=self.bases,
                             # activation='softmax')([kb_encoder] + A_in)

        
        # A' - answer decoder
        answer_decoder = []
        for i in range(self.decoder_depth):
            answer_decoder.append(GRU(
                self.rnn_units, 
                return_state=True,
                return_sequences=True, 
                name='answer_decoder_%i'%i,
                ))

        decoder_softmax = Dense(num_entities + 1,        # +1 as mask_zero
                                activation='softmax', name='decoder_softmax')

        # network architecture
        question_encoder_output, question_encoder_states = self._stacked_rnn(
                question_encoder, word_embedding(question_encoder_input))

        decoder_outputs, decoder_states = self._stacked_rnn(answer_decoder, kb_encoder_output, [question_encoder_states[-1]] * self.decoder_depth)
        decoder_outputs = Dropout(self.dropout_rate)(decoder_outputs)
        decoder_outputs = decoder_softmax(decoder_outputs)

        self.model_train = Model(
                [question_encoder_input, kb_encoder_input],   # [input question, input KB],
                answer_decoder_output)                        # ground-truth target answer

        def train(self, batch_size, epochs, batch_per_load=10, lr=0.001):
            '''
            '''
            self.model_train.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy')

            for epoch in range(epochs):
                # load training dataset
                # encoder_input_data, decoder_input_data, decoder_target_data, _, _ = self.dataset.load_data('train', batch_size * batch_per_load)

                self.model_train.fit(
                    [question_encoder_input_data, kb_encoder_input_data], 
                     answer_decoder_target_data, batch_size=batch_size,)

                self.save_model('model_epoch%i.h5'%(epoch + 1))
            self.save_model('model.h5')


if __name__ == '__main__':

    # load embeddings
    if not os.path.exists(EMBEDDINGS_PATH):
        os.makedirs(EMBEDDINGS_PATH)

    if not os.path.exists(GLOVE_EMBEDDINGS_PATH):
        wget.download('http://nlp.stanford.edu/data/glove.6B.zip', EMBEDDINGS_PATH+"glove.6B.zip")
        with zipfile.ZipFile(EMBEDDINGS_PATH+"glove.6B.zip","r") as zip_ref:
            zip_ref.extractall(EMBEDDINGS_PATH)

    # define QA model architecture parameters
    rnn_units = 512
    encoder_depth = 2
    decoder_depth = 2
    dropout_rate = 0.5

    # define R-GCN architecture parameters
    num_hidden_units = 16
    bases = -1
    l2norm = 0.

    # define training parameters
    batch_size = 100
    epochs = 10
    learning_rate = 1e-3

    # initialize the model
    model = KBQA_RGCN(rnn_units, encoder_depth, decoder_depth, num_hidden_units, bases, l2norm, dropout_rate)
    
    # set mode
    mode = 'train'
    
    # modes
    if mode == 'train':
        model.build_model_train()
        model.train(batch_size, epochs, lr=learning_rate)
