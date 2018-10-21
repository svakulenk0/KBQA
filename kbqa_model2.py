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

from keras.layers import Input, GRU, Dropout, Embedding, Dot
from keras.callbacks import  ModelCheckpoint, EarlyStopping
from keras.regularizers import l2
from keras.optimizers import Adam
from keras import backend as K

from utils import *
from kbqa2_settings import *

class KBQA2:
    '''
    Second neural network architecture for KBQA: projecting from word and KG embeddings aggregation into the KG answer space
    '''

    def __init__(self, rnn_units, train_word_embeddings, train_kg_embeddings, model_path='./models/model.best.hdf5', load_word_embeddings=readGloveFile):
        # define path to store pre-trained model
        makedirs('./models')
        self.model_path = model_path

        # set architecture parameters
        self.rnn_units = rnn_units
        self.train_word_embeddings = train_word_embeddings
        self.train_kg_embeddings = train_kg_embeddings

        # load word embeddings with its vocabulary into maps
        self.wordToIndex, self.indexToWord, self.wordToGlove = load_word_embeddings()
        self.num_words = len(self.wordToIndex.keys())
        print("Number of words with pre-trained embeddings: %d"%self.num_words)

        # load KG embeddings
        self.entityToIndex, self.indexToEntity, self.entityToVec, kb_embeddings_dimension = load_KB_embeddings()
        self.num_entities = len(self.entityToIndex.keys())
        print("Number of entities with pre-trained embeddings: %d"%self.num_entities)

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
        questions_data = np.asarray(pad_sequences(questions_data, padding='post'), dtype=K.floatx())
        answers_data = np.asarray(answers_data, dtype=K.floatx())
        self.num_samples = answers_data.shape[0]

        print("Loaded the dataset")
        self.dataset = questions_data, answers_data

        # show dataset stats
        print("Number of samples: %d"%self.num_samples)
        print("Maximum number of words in a question sequence: %d"%questions_data.shape[1])

        if show_n_answers_distribution:
            print("Number of answers per question distribution: %s"%str(n_answers_per_question))

    def build_model(self):
        '''
        build layers required for training the NN
        '''

        # load word & KG embeddings
        word_embeddings_matrix = load_embeddings_from_index(self.wordToGlove, self.wordToIndex)
        kg_embeddings_matrix = load_embeddings_from_index(self.entityToVec, self.entityToIndex)

        # Q - question input
        question_input = Input(shape=(None,), name='question_input', dtype=K.floatx())

        # E' - question words embedding: set up a trainable word embeddings layer initialized with pre-trained word embeddings
        question_words_embeddings = Embedding(word_embeddings_matrix.shape[0], word_embeddings_matrix.shape[1],
                                     weights=[word_embeddings_matrix], trainable=self.train_word_embeddings,
                                     name='question_words_embeddings', mask_zero=True)
        question_embedding_output = question_words_embeddings(question_input)

        # Q' - question encoder
        question_encoder_output_1 = GRU(self.rnn_units, name='question_encoder_1', return_sequences=True)(question_embedding_output)
        question_encoder_output_2 = GRU(self.rnn_units, name='question_encoder_2', return_sequences=True)(question_encoder_output_1)
        question_encoder_output_3 = GRU(self.rnn_units, name='question_encoder_3', return_sequences=True)(question_encoder_output_2)
        question_encoder_output_4 = GRU(self.rnn_units, name='question_encoder_4', return_sequences=True)(question_encoder_output_3)
        question_encoder_output = GRU(kg_embeddings_matrix.shape[1], name='question_encoder')(question_encoder_output_4)


        # E'' - KG entity embeddings: load pre-trained vectors e.g. RDF2vec, as constant/variable ?
        kg_embeddings = K.variable(kg_embeddings_matrix)

        # A - answer output as the dot product between the question vector and the KG embeddings
        answer_embedding_output = Dot(axes=1, normalize=True)([question_encoder_output, kg_embeddings])
        # answers_output = Dense(axes=1, normalize=True)([question_encoder_output, kg_embeddings])

        # A - answer decoder
        answer_decoder_output_1 = GRU(self.rnn_units, name='question_encoder_1', return_sequences=True)(answer_embedding_output)
        answer_decoder_output_2 = GRU(self.rnn_units, name='question_encoder_2', return_sequences=True)(answer_decoder_output_1)
        answer_decoder_output_3 = GRU(self.rnn_units, name='question_encoder_3', return_sequences=True)(answer_decoder_output_2)
        answer_decoder_output_4 = GRU(self.rnn_units, name='question_encoder_4', return_sequences=True)(answer_decoder_output_3)
        answer_decoder_output = GRU(self.num_entities, name='answer_decoder')(answer_decoder_output_4)

        self.model_train = Model(inputs=[question_input],   # input question
                                 outputs=[answer_decoder_output])  # ground-truth target answer set
        print(self.model_train.summary())

    def train(self, batch_size, epochs, lr=0.001):
        # define loss
        self.model_train.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', metrics=['accuracy'])

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

    model = KBQA2(rnn_units, train_word_embeddings, train_kg_embeddings)
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
