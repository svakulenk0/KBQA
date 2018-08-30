#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Aug 30, 2018

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Neural network model with R-GCN layer for KBQA

Based on Keras implementation of RGCN layer https://github.com/tkipf/relational-gcn

Question - text as the sequence of words (word index)
Answer - entity from KB (entity index)

'''
import sys

from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

from keras.models import Model
from keras.models import load_model

from keras.layers import Input, GRU, Dropout, Embedding, Dense, Flatten
from keras.callbacks import  ModelCheckpoint, EarlyStopping
from keras.regularizers import l2
from keras.optimizers import Adam
from keras import backend as K

from rgcn.layers.graph import GraphConvolution
from rgcn.layers.input_adj import InputAdj

from utils import *
from rgcn_settings import *


class KBQA_RGCN:
    '''
    NN model for KBQA with R-GCN for KB embedding training
    '''
    def __init__(self, rnn_units, gc_units, gc_bases, l2norm, train_word_embeddings, load_word_embeddings=readGloveFile):

        # set architecture parameters
        self.rnn_units = rnn_units
        self.gc_units = gc_units
        self.gc_bases = gc_bases
        self.l2norm = l2norm
        self.train_word_embeddings = train_word_embeddings

        # load word embeddings with its vocabulary into maps
        self.wordToIndex, self.indexToWord, self.wordToGlove = load_word_embeddings()
        self.num_words = (len(self.wordToIndex.keys()))

        # load entity vocabulary into a map
        self.entityToIndex, self.kb_adjacency = loadKB()
        self.num_entities = len(self.entityToIndex.keys())
        self.support = len(self.kb_adjacency)  # number of relations in KB?

    def load_data(self, dataset):
        questions, answers = dataset
        num_samples = len(questions)
        assert num_samples == len(answers)

        # encode questions with word vocabulary and answers with entity vocabulary
        questions_data = []
        answers_data = []

        # iterate over samples
        for i in range(num_samples):
            # encode words in the question (ignore OOV words i.e. words without pre-trained embeddings) TODO: deal with OOV e.g. char-based encoding or FastText
            questions_sequence = [self.wordToIndex[word] for word in text_to_word_sequence(questions[i]) if word in self.wordToIndex]
            questions_data.append(questions_sequence)

            # encode all entities in the answer (ignore OOV entity labels i.e. entities in the answers but not in the KB)
            answer_set = [self.entityToIndex[entity] for entity in answers[i] if entity in self.entityToIndex]
            answers_data.append(answer_set)

        # normalize length
        questions_data = np.asarray(pad_sequences(questions_data, padding='post'))
        answers_data = np.asarray(pad_sequences(answers_data, padding='post'))

        print("Loaded the dataset")
        # show dataset stats
        print("Maximum number of words in a question sequence: %d"%questions_data.shape[1])
        print("Maximum number of entities in an answer set: %d"%answers_data.shape[1])

    def build_model_train(self):
        '''
        build layers required for training the NN
        '''
        # set up a trainable word embeddings layer initialized with pre-trained word embeddings
        embeddings_matrix = load_embeddings_from_index(self.wordToGlove, self.wordToIndex)
        words_embeddings = Embedding(embeddings_matrix.shape[0], embeddings_matrix.shape[1],
                                     weights=[embeddings_matrix], trainable=self.train_word_embeddings,
                                     name='words_embeddings', mask_zero=True)

        # Q - question input
        question_input = Input(shape=(None,), name='question_input')

        # E' - question words embedding
        question_embedding_output = words_embeddings(question_input)

        # Q' - question encoder
        question_encoder_output_1 = GRU(self.rnn_units, name='question_encoder_1', return_sequences=True)(question_embedding_output)
        question_encoder_output_2 = GRU(self.rnn_units, name='question_encoder_2', return_sequences=True)(question_encoder_output_1)
        question_encoder_output_3 = GRU(self.rnn_units, name='question_encoder_3', return_sequences=True)(question_encoder_output_2)
        question_encoder_output_4 = GRU(self.rnn_units, name='question_encoder_4', return_sequences=True)(question_encoder_output_3)
        question_encoder_output = GRU(self.gc_units, name='question_encoder')(question_encoder_output_4)

        # K - KB input: entities as sequences of words and relations as adjacency matrix
        # https://github.com/tkipf/relational-gcn
        kb_adjacency_input = [InputAdj() for _ in range(self.support)]
        kb_entities_input = Input(shape=(self.num_entities,))

        # kb_entities_input = Input(shape=(None,), name='kb_entities_input')
        
        # E'' - KB entity embedding for entity labels using the same pre-trained word embeddings
        # kb_entities_words_embedding_output = words_embeddings(kb_entities_input)
        # # aggregate word embeddings vectors into a single entity vector
        kb_entities_embedding_output = kb_entities_input

        kb_input = [kb_entities_embedding_output] + kb_adjacency_input

        # K' - KB encoder layer via R-GCN
        # https://github.com/tkipf/relational-gcn
        kb_encoder_output = GraphConvolution(self.gc_units, self.support, num_bases=self.gc_bases, featureless=False,
                                             activation='relu',
                                             W_regularizer=l2(self.l2norm))(kb_input)

        # S' - KB subgraph projection layer
        kb_projection_output = Dot()([kb_encoder_output, question_encoder_output])

        # A - answer output
        answers_output = Dense(self.num_entities, activation="sigmoid")(kb_projection_output)

        self.model_train = Model(inputs=[question_input, kb_adjacency_input, kb_entities_input],   # input question TODO input KB
                                 outputs=[answers_output])  # ground-truth target answer set
        print self.model_train.summary()

    def train(self, batch_size, epochs, lr=0.001):
        # define loss
        self.model_train.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy')

        # define callbacks for early stopping
        checkpoint = ModelCheckpoint(self.model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min') 
        callbacks_list = [checkpoint, early_stop]
        
        # prepare QA dataset and KB
        questions_vectors, answers_vectors = self.dataset
        # represent KB entities with 1-hot encoding vectors
            # kb_entities = sp.csr_matrix(self.kb_adjacency[0].shape)
        kb_entities = K.eye(self.kb_adjacency[0].shape[0])

        self.model_train.fit([questions_vectors, self.kb_adjacency, kb_entities], [answers_vectors], epochs=epochs, callbacks=callbacks_list, verbose=2, validation_split=0.3, shuffle='batch')


def main(mode):
    '''
    Train model by running: python rgcn_kbqa2.py train
    '''
    # from rgcn_settings import dataset_name, rnn_units, gc_units, gc_bases, l2norm, train_word_embeddings, batch_size, epochs, learning_rate

    model = KBQA_RGCN(rnn_units, gc_units, gc_bases, l2norm, train_word_embeddings)
    # train on train split / test on test split
    dataset_split = mode

    # load data
    dataset = load_dataset(dataset_name, dataset_split)
    model.load_data(dataset)

    # mode switch
    if mode == 'train':
        # build model
        model.build_model_train()
        # train model
        model.train(batch_size, epochs, lr=learning_rate)
    # elif mode == 'test':
    #     model.load_pretrained_model()
    #     print("Loaded the pretrained model")
    #     model.test()


if __name__ == '__main__':
    set_random_seed()
    main(sys.argv[1])
