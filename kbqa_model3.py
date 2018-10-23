#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Oct 18, 2018

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Neural network model with pre-trained KG graph embeddings layer for KBQA

Question - text as the sequence of words (word index)
Answer - entity from KB (entity index)

'''
import sys
from collections import Counter

from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

from keras.models import Model
from keras.models import load_model

from keras.layers import Input, GRU, Dropout, Embedding, Lambda
from keras.callbacks import  ModelCheckpoint, EarlyStopping
from keras.regularizers import l2
from keras.optimizers import Adam
from keras import backend as K

from utils import *
from kbqa3_settings import *


class KBQA:
    '''
    Second neural network architecture for KBQA: projecting from word and KG embeddings aggregation into the KG answer space
    '''

    def __init__(self, rnn_units, model_path='./models/model.best.hdf5'):
        # define path to store pre-trained model
        makedirs('./models')
        self.model_path = model_path

        # set architecture parameters
        self.rnn_units = rnn_units
        # self.train_word_embeddings = train_word_embeddings
        self.train_kg_embeddings = train_kg_embeddings

        # load word embeddings model
        self.wordToVec = load_fasttext()
        self.word_embs_dim = len(self.wordToVec.get_word_vector('sample'))
        print("FastText word embeddings dimension: %d"%self.word_embs_dim)

        # load KG relation embeddings
        self.entityToIndex, self.indexToEntity, self.entityToVec, self.kb_embeddings_dimension = load_KB_embeddings()
        self.num_entities = len(self.entityToIndex.keys())
        print("Number of entities with pre-trained embeddings: %d"%self.num_entities)
        self.kg_relation_embeddings_matrix = load_embeddings_from_index(self.entityToVec, self.entityToIndex)

        # generate KG word embeddings
        kg_word_embeddings_matrix = np.zeros((self.num_entities+1, self.word_embs_dim))  # initialize with zeros (adding 1 to account for masking)
        for entity_id, index in self.entityToIndex.items():
            # print index, entity_id
            kg_word_embeddings_matrix[index, :] = self.wordToVec.get_word_vector(entity_id) # create embedding: item index to item embedding
        self.kg_word_embeddings_matrix = np.asarray(kg_word_embeddings_matrix, dtype=K.floatx())

    def load_data(self, dataset, max_answers_per_question=100, show_n_answers_distribution=False):
        '''
        Encode the dataset: questions and answers
        '''
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
            # encode words in the question using FastText
            question_word_vectors = [self.wordToVec.get_word_vector(word) for word in text_to_word_sequence(questions[i])]

            # encode all entities in the answer as a list of indices (ignore OOV entity labels i.e. entities in the answers but not in the KB)
            answer_set = [self.entityToIndex[entity] for entity in answers[i] if entity in self.entityToIndex]
            n_answers = len(answer_set)

            # add sample that has not too many answers but at least one
            if answer_set and n_answers <= max_answers_per_question:
                # print answer_set
                n_answers_per_question[n_answers] += 1
                questions_data.append(question_word_vectors)

                # encode all entities in the answer as a one-hot-vector for the corresponding entities indices +1 for 0 index which is a mask
                answer_vector = np.zeros(self.num_entities+1)
                answer_vector[answer_set] = 1
                answers_data.append(answer_vector)

        # normalize length
        questions_data = np.asarray(pad_sequences(questions_data, padding='post'), dtype=K.floatx())
        answers_data = np.asarray(answers_data, dtype=K.floatx())
        
        self.num_samples = questions_data.shape[0]
        self.max_question_words = questions_data.shape[1]

        print("Loaded the dataset")
        self.dataset = questions_data, answers_data

        # show dataset stats
        print("Number of samples: %d"%self.num_samples)
        # print(questions_data.shape)
        print("Maximum number of words in a question sequence: %d"%self.max_question_words)

        if show_n_answers_distribution:
            print("Number of answers per question distribution: %s"%str(n_answers_per_question))

        # check the input data 
        # print questions_data
        # print answers_data

    def dot_product_layer(self, tensor1, tensor2):
        '''
        Custom layer producing a dot product
        '''
        return K.dot(tensor1, tensor2)

    def build_model(self):
        '''
        build layers required for training the NN
        '''

        # Q - question embedding input
        question_embeddings_input = Input(shape=(self.max_question_words, self.word_embs_dim), name='question_embedding_input', dtype=K.floatx())

        # K - KG word embeddings
        kg_word_embeddings = K.constant(self.kg_word_embeddings_matrix.T)
        
        # S - selected KG entities
        selected_entities = Lambda(self.dot_product_layer, name='selected_entities')(question_embeddings_input, kg_word_embeddings)

        # R - KG relation embeddings
        kg_relation_embeddings = K.constant(self.kg_relation_embeddings_matrix.T)

        # S' - selected KG entities with relations subgraph
        # selected_subgraph = 

        # A - answer entities output decoder
        # answer_entities_output =

        self.model_train = Model(inputs=[question_embeddings_input],   # input question
                                 outputs=[answer_entities_output])  # ground-truth target answer set
        print(self.model_train.summary())

    def train(self, batch_size, epochs, lr=0.001):
        # define loss
        self.model_train.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])

        # define callbacks for early stopping
        checkpoint = ModelCheckpoint(self.model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min') 
        callbacks_list = [early_stop]
        
        # prepare QA dataset
        questions_vectors, answers_vectors = self.dataset

        self.model_train.fit([questions_vectors], [answers_vectors], epochs=epochs, callbacks=callbacks_list, verbose=1, validation_split=0.3, shuffle='batch', batch_size=batch_size)


def main(mode):
    '''
    Train model by running: python kbqa_model2.py train
    '''

    model = KBQA(rnn_units)
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


if __name__ == '__main__':
    set_random_seed()
    main(sys.argv[1])
