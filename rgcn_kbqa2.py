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
from collections import Counter

from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

from keras.models import Model
from keras.models import load_model

from keras.layers import Input, GRU, Dropout, Embedding, Dense, Flatten, Dot
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
    def __init__(self, rnn_units, gc_units, gc_bases, l2norm, train_word_embeddings, model_path='./models/model.best.hdf5', load_word_embeddings=readGloveFile):
        # define path to store pre-trained model
        makedirs('./models')
        self.model_path = model_path

        # set architecture parameters
        self.rnn_units = rnn_units
        self.gc_units = gc_units
        self.gc_bases = gc_bases
        self.l2norm = l2norm
        self.train_word_embeddings = train_word_embeddings

        # load word embeddings with its vocabulary into maps
        self.wordToIndex, self.indexToWord, self.wordToGlove = load_word_embeddings()
        self.num_words = (len(self.wordToIndex.keys()))
        print("Number of words in vocabulary with pre-trained embeddings: %d"%self.num_words)

        # load entity vocabulary into a map
        self.entityToIndex, self.kb_adjacency = loadKB()
        self.num_entities = len(self.entityToIndex.keys())
        print("Number of entities in KB vocabulary: %d"%self.num_entities)
        self.support = len(self.kb_adjacency)  # number of relations in KB?
        print("Number of relations in KB adjacency matrix: %d"%self.support)

    def load_data(self, dataset, max_answers_per_question=100, show_n_answers_distribution=False):
        questions, answers = dataset
        num_samples = len(questions)
        assert num_samples == len(answers)

        # encode questions with word vocabulary and answers with entity vocabulary
        questions_data = []
        answers_data = []
        # track number of answers per question distribution
        n_answers_per_question = Counter()

        # iterate over samples
        for i in range(num_samples):
            # encode words in the question (ignore OOV words i.e. words without pre-trained embeddings) TODO: deal with OOV e.g. char-based encoding or FastText
            questions_sequence = [self.wordToIndex[word] for word in text_to_word_sequence(questions[i]) if word in self.wordToIndex]

            # encode all entities in the answer as a list of indices (ignore OOV entity labels i.e. entities in the answers but not in the KB)
            answer_set = [self.entityToIndex[entity] for entity in answers[i] if entity in self.entityToIndex]
            n_answers = len(answer_set)

            # add sample
            if n_answers <= max_answers_per_question:
                n_answers_per_question[n_answers] += 1
                questions_data.append(questions_sequence)

                # encode all entities in the answer as a one-hot-vector for the corresponding entities indices TODO
                answer_vector = np.zeros(self.num_entities)
                answer_vector[answer_set] = 1
                answers_data.append(answer_vector)

        # normalize length
        questions_data = np.asarray(pad_sequences(questions_data, padding='post'))
        print("Number of samples: %d"%len(answers_data))
        answers_data = np.asarray(answers_data)

        print("Loaded the dataset")
        self.dataset = questions_data, answers_data

        # show dataset stats
        print("Maximum number of words in a question sequence: %d"%questions_data.shape[1])
        print("Maximum number of entities in an answer set: %d"%answers_data.shape[1])

        if show_n_answers_distribution:
            print("Number of answers per question distribution: %s"%str(n_answers_per_question))

    def build_model(self):
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

        # test
        # A = K.variable(self.kb_adjacency[0])
        # X = K.random_uniform_variable(shape=(self.num_entities, 4), low=0, high=1)
        # sparse_matrix = K.dot(A, X)

        kb_adjacency_input = [K.variable(kb_relation_adjacency) for kb_relation_adjacency in self.kb_adjacency]
        # kb_adjacency_input = [kb_relation_adjacency for kb_relation_adjacency in self.kb_adjacency]
        # represent KB entities with 1-hot encoding vectors
            # kb_entities = sp.csr_matrix(self.kb_adjacency[0].shape)
        kb_entities_input = Input(shape=(self.num_entities,))
        
        # E'' - KB entity embedding for entity labels using the same pre-trained word embeddings
        # kb_entities_words_embedding_output = words_embeddings(kb_entities_input)
        # # aggregate word embeddings vectors into a single entity vector
        kb_entities_embedding_output = kb_entities_input

        # kb_input = [kb_entities_embedding_output] + kb_adjacency_input

        # input_tensor = K.placeholder(shape=self.num_entities,
        #                              dtype=K.floatx(),
        #                              sparse=True)

        # K' - KB encoder layer via R-GCN
        # https://github.com/tkipf/relational-gcn
        kb_encoder_output = GraphConvolution(self.gc_units, kb_adjacency_input, self.support, num_bases=self.gc_bases, featureless=True,
                                             activation='relu', W_regularizer=l2(self.l2norm))(kb_entities_embedding_output)

        # S' - KB subgraph projection layer
        # check tensor shapes before multiplication
        # print("Question encoder output shape: %s"%str(K.shape(question_encoder_output)))
        # print("KB encoder output shape: %s"%str(K.shape(kb_encoder_output)))
        kb_projection_output = Dot(axes=1, normalize=True)([question_encoder_output, kb_encoder_output])
        # kb_projection_output = K.dot(question_encoder_output, kb_encoder_output)

        # A - answer output
        answers_output = Dense(self.num_entities, activation="sigmoid")(kb_projection_output)

        self.model_train = Model(inputs=[question_input, kb_entities_input],   # input question TODO input KB
                                 outputs=[answers_output])  # ground-truth target answer set
        print self.model_train.summary()

    def train(self, batch_size, epochs, lr=0.001):
        # define loss
        self.model_train.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy')

        # define callbacks for early stopping
        checkpoint = ModelCheckpoint(self.model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min') 
        callbacks_list = [checkpoint, early_stop]
        
        # prepare QA dataset
        questions_vectors, answers_vectors = self.dataset
        kb_entities = K.random_uniform_variable(shape=(self.num_entities, 4), low=0, high=1)

        self.model_train.fit([questions_vectors, kb_entities], [answers_vectors], epochs=epochs, callbacks=callbacks_list, verbose=2, validation_split=0.3, shuffle='batch')


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
        model.build_model()
        # train model
        model.train(batch_size, epochs, lr=learning_rate)
    # elif mode == 'test':
    #     model.load_pretrained_model()
    #     print("Loaded the pretrained model")
    #     model.test()


if __name__ == '__main__':
    set_random_seed()
    main(sys.argv[1])
