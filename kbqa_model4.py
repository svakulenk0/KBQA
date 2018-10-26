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

from sklearn.metrics.pairwise import cosine_similarity

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
from kbqa_settings import *


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
        # self.n_words = n_words  # maximum number of words in a question

        # self.train_word_embeddings = train_word_embeddings
        # self.train_kg_embeddings = train_kg_embeddings

        # load word embeddings model
        self.wordToVec = load_fasttext()
        self.word_embs_dim = len(self.wordToVec.get_word_vector('sample'))
        print("FastText word embeddings dimension: %d"%self.word_embs_dim)

        # load KG relation embeddings
        self.entityToIndex, self.indexToEntity, self.entityToVec, self.kb_embeddings_dimension = load_KB_embeddings()
        self.entities = self.entityToVec.keys()
        self.num_entities = len(self.entityToIndex.keys())
        print("Number of entities with pre-trained embeddings: %d"%self.num_entities)
        self.kg_relation_embeddings_matrix = load_embeddings_from_index(self.entityToVec, self.entityToIndex)
        print("RDF2Vec embeddings dimension: %d"%self.kb_embeddings_dimension)

        # generate KG word embeddings
        kg_word_embeddings_matrix = np.zeros((self.num_entities+1, self.word_embs_dim))  # initialize with zeros (adding 1 to account for masking)
        for entity_id, index in self.entityToIndex.items():
            # print index, entity_id
            kg_word_embeddings_matrix[index, :] = self.wordToVec.get_word_vector(entity_id) # create embedding: item index to item embedding
        self.kg_word_embeddings_matrix = np.asarray(kg_word_embeddings_matrix, dtype=K.floatx())

    def load_data(self, mode, max_question_words=None, max_answers_per_question=100):
        '''
        Encode the dataset: questions and answers
        '''
        # load data
        questions, answers = load_dataset(dataset_name, dataset_split=mode)
        num_samples = len(questions)
        assert num_samples == len(answers)

        # encode questions with word vocabulary and answers with entity vocabulary
        question_vectors = []
        answer_vectors = []
        # evaluating against all correct answers at test time
        all_answers_indices = []

        # iterate over QA samples
        for i in range(num_samples):
            # evaluating against all correct answers at test time
            correct_answers = []
            # train on one answer only
            sample_answer = False
            for answer in answers[i]:
                answer = answer.encode('utf-8')
                if answer in self.entities:
                    if not sample_answer:
                        answer_vectors.append(self.entityToVec[answer])
                        # encode words in the question using FastText
                        question_vectors.append([self.wordToVec.get_word_vector(word) for word in text_to_word_sequence(questions[i])])
                        sample_answer = True

                    correct_answers.append(self.entityToIndex[answer])

            # skip questions that have no answer embeddings
            if correct_answers:
                all_answers_indices.append(correct_answers)

        # normalize input length
        if max_question_words:
            # pad to the size of the trained model
            question_vectors = np.asarray(pad_sequences(question_vectors, padding='post', maxlen=max_question_words))
            print("Maximum question length %d padded to %d"%(question_vectors.shape[1], max_question_words))
        else:
            # get the max size on the training set
            question_vectors = np.asarray(pad_sequences(question_vectors, padding='post'), dtype=K.floatx())
            self.max_question_words = question_vectors.shape[1]
            print("Maximum number of words in a question sequence: %d"%self.max_question_words)

        answer_vectors = np.asarray(answer_vectors, dtype=K.floatx())
        
        self.num_samples = question_vectors.shape[0]
        print("Number of samples: %d"%self.num_samples)

        print("Loaded the dataset")
        self.dataset = (question_vectors, answer_vectors, all_answers_indices)

    def entity_linking_layer(self, question_vector):
        '''
        Custom layer producing a dot product
        '''
        # K - KG embeddings
        kg_embeddings = K.constant(self.kg_word_embeddings_matrix.T)

        return K.dot(question_vector, kg_embeddings)

    def kg_embedding_layer(self, selected_entities):
        '''
        Custom layer adding matrix to a tensor
        '''
        # R - KG relation embeddings
        kg_relation_embeddings = K.constant(self.kg_relation_embeddings_matrix)

        return K.dot(selected_entities, kg_relation_embeddings)

    def build_model(self):
        '''
        build layers required for training the NN
        '''

        # Q - question embedding input
        question_input = Input(shape=(self.max_question_words, self.word_embs_dim), name='question_input', dtype=K.floatx())

        # S - selected KG entities
        selected_entities = Lambda(self.entity_linking_layer, name='selected_entities')(question_input)

        # S' - selected KG subgraph
        # selected_subgraph = Lambda(self.kg_embedding_layer, name='selected_subgraph')(selected_entities)

        # A - answer decoder
        answer_decoder_1 = GRU(self.rnn_units, name='answer_decoder_1', return_sequences=True)(selected_subgraph)
        answer_decoder_2 = GRU(self.rnn_units, name='answer_decoder_2', return_sequences=True)(answer_decoder_1)
        answer_decoder_3 = GRU(self.rnn_units, name='answer_decoder_3', return_sequences=True)(answer_decoder_2)
        answer_decoder_4 = GRU(self.rnn_units, name='answer_decoder_4', return_sequences=True)(answer_decoder_3)
        answer_output = GRU(self.kb_embeddings_dimension, name='answer_output')(answer_decoder_4)

        self.model_train = Model(inputs=[question_input],   # input question
                                 outputs=[answer_output])  # ground-truth target answer set
        print(self.model_train.summary())

    def train(self, batch_size, epochs, lr=0.001):
        # define loss
        self.model_train.compile(optimizer=Adam(lr=lr), loss='cosine_proximity')

        # define callbacks for early stopping
        checkpoint = ModelCheckpoint(self.model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min') 
        callbacks_list = [checkpoint, early_stop]
        
        question_vectors, answer_vectors, all_answers_indices = self.dataset

        self.model_train.fit([question_vectors], [answer_vectors], epochs=epochs, callbacks=callbacks_list, verbose=2, validation_split=0.3, shuffle='batch', batch_size=batch_size)

    def test(self):
        '''
        '''
        self.model_train = load_model(self.model_path, custom_objects={'entity_linking_layer': self.entity_linking_layer,
                                                                       'kg_embedding_layer': self.kg_embedding_layer})
        print("Loaded the pre-trained model")

        question_vectors, answer_vectors, all_answers_indices = self.dataset
        print("Testing...")
        # score = self.model_train.evaluate(questions, answers, verbose=0)
        # print score
        print("Questions vectors shape: " + " ".join([str(dim) for dim in question_vectors.shape]))
        # print("Answers vectors shape: " + " ".join([str(dim) for dim in answers_vectors.shape]))
        print("Answers indices shape: %d" % len(all_answers_indices))

        predicted_answers_vectors = self.model_train.predict(question_vectors)
        print("Predicted answers vectors shape: " + " ".join([str(dim) for dim in predicted_answers_vectors.shape]))
        # print("Answers indices: " + ", ".join([str(idx) for idx in answers_indices]))

        # calculate pairwise distances (via cosine similarity)
        similarity_matrix = cosine_similarity(predicted_answers_vectors, self.kg_relation_embeddings_matrix)

        # print np.argmax(similarity_matrix, axis=1)
        n = 5
        # indices of the top n predicted answers for every question in the test set
        top_ns = similarity_matrix.argsort(axis=1)[:, -n:][::-1]
        # print top_ns[:2]

        hits = 0
        for i, answers in enumerate(all_answers_indices):
            # check if the correct and predicted answer sets intersect
            if set.intersection(set(answers), set(top_ns[i])):
            # if set.intersection(set([answers[0]]), set(top_ns[i])):
                hits += 1

        print("Hits in top %d: %d/%d"%(n, hits, len(all_answers_indices)))


def main(mode):
    '''
    Train model by running: python kbqa_modeli.py train
    '''

    model = KBQA(rnn_units)

    # mode switch
    if mode == 'train':
        model.load_data(mode)
        # build model
        model.build_model()
        # train model
        model.train(batch_size, epochs, lr=learning_rate)
    
    elif mode == 'test':
        model.load_data(mode, max_question_words)
        model.test()


if __name__ == '__main__':
    set_random_seed()
    main(sys.argv[1])
