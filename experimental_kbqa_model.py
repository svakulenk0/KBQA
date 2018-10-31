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

from keras.layers import Input, GRU, Dropout, Embedding, Lambda, Dense
from keras.callbacks import  ModelCheckpoint, EarlyStopping
from keras.regularizers import l2
from keras.optimizers import SGD, Adam, Nadam, Adadelta

from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.utils import CustomObjectScope

from utils import *
from kbqa_settings import *
from EL_layer import EntityLinking
from lcquad_train_balanced import lcquad_train_b


class KBQA:
    '''
    Second neural network architecture for KBQA: projecting from word and KG embeddings aggregation into the KG answer space
    '''

    def __init__(self, rnn_units, output_vector, model_path='./models/model.best.hdf5'):
        # define path to store pre-trained model
        makedirs('./models')
        self.model_path = model_path

        self.dataset = None

        # set architecture parameters
        self.rnn_units = rnn_units
        self.output_vector = output_vector
        # self.n_words = n_words  # maximum number of words in a question

        # self.train_word_embeddings = train_word_embeddings
        # self.train_kg_embeddings = train_kg_embeddings

        # load word embeddings model
        self.wordToVec = load_fasttext()
        self.word_embs_dim = len(self.wordToVec.get_word_vector('sample'))
        print("FastText word embeddings dimension: %d"%self.word_embs_dim)

        # load KG relation embeddings
        self.entityToIndex, self.indexToEntity, self.entityToVec, self.kg_relation_embeddings_matrix = load_KB_embeddings()

        self.entities = self.entityToIndex.keys()
        self.num_entities = len(self.entities)
        assert len(self.entityToVec.keys()) == self.num_entities
        
        assert self.kg_relation_embeddings_matrix.shape[0] == self.num_entities
        print("Number of entities with pre-trained embeddings: %d"%self.num_entities)

        self.kg_embeddings_dim = self.kg_relation_embeddings_matrix.shape[1]
        assert self.kg_embeddings_dim == len(self.entityToVec[0])
        print("KG embeddings dimension: %d"%self.kg_relation_embeddings_matrix.shape[1])

        # generate KG word embeddings
        kg_word_embeddings_matrix = np.zeros((self.num_entities, self.word_embs_dim))  # initialize with zeros (adding 1 to account for masking)
        for entity_id, index in self.entityToIndex.items():
            # print index, entity_id
            # entity = entity_id.split('/')[-1].split('_(')[0]
            entity = entity_id.split('/')[-1]
            # print entity
            kg_word_embeddings_matrix[index, :] = self.wordToVec.get_word_vector(entity) # create embedding: item index to item embedding
        self.kg_word_embeddings_matrix = np.asarray(kg_word_embeddings_matrix, dtype=K.floatx())
        
        # self.kg_embeddings_matrix = np.dot(self.kg_word_embeddings_matrix.T, self.kg_relation_embeddings_matrix)

    def load_data(self, dataset_name, split, max_question_words=None, max_answers_per_question=100, balance=True):
        '''
        Encode the dataset: questions and answers
        '''
        # load data
        questions, answers = load_dataset(dataset_name, split)
        num_samples = len(questions)
        assert num_samples == len(answers)
        print('Loaded %s with %d %s QA samples' % (dataset_name, num_samples, split))

        if balance:
            # filter out questions with frequent answers using projection to the indices mask
            balanced_question = []
            balanced_answers = []
            for idx in lcquad_train_b:
                balanced_question.append(questions[idx])
                balanced_answers.append(answers[idx])
            # replace the original dataset with the filtered/balanced one
            questions, answers = balanced_question, balanced_answers
            num_samples = len(questions)
            assert num_samples == len(answers)
            print('Rebalanced to %d QA samples in the new %s dataset from %s' % (num_samples, split, dataset_name))

        # encode questions with word vocabulary and answers with entity vocabulary
        question_vectors = []
        answer_vectors = []
        # evaluating against all correct answers at test time
        all_answers_indices = []
        # iterate over QA samples
        for i in range(num_samples):
            # evaluating against all correct answers at test time
            correct_answers = []
            for answer in answers[i]:
                answer = answer.encode('utf-8')
                # consider only samples where we can embed the answer
                if answer in self.entities:
                    correct_answers.append(self.entityToIndex[answer])

            # skip questions that have no answer embeddings
            if correct_answers:
                # encode words in the question using FastText
                question_vectors.append([self.wordToVec.get_word_vector(word) for word in text_to_word_sequence(questions[i])])
                all_answers_indices.append(correct_answers)

        # normalize input length
        if max_question_words:
            # at test time: pad to the size of the trained model
            question_vectors = np.asarray(pad_sequences(question_vectors, padding='post', maxlen=max_question_words))
            print("Maximum question length %d padded to %d"%(question_vectors.shape[1], max_question_words))
        else:
            # at training time: get the max size for this training set
            question_vectors = np.asarray(pad_sequences(question_vectors, padding='post'), dtype=K.floatx())
            self.max_question_words = question_vectors.shape[1]
            print("Maximum number of words in a question sequence: %d"%self.max_question_words)

        # train on the first available answer only
        first_answers = [answers[0] for answers in all_answers_indices]
        if self.output_vector == 'one-hot':
            answer_vectors = to_categorical(first_answers, num_classes=self.num_entities)
        elif self.output_vector == 'embedding':
            answer_vectors = [self.entityToVec[answer] for answer in first_answers]
            answer_vectors = np.asarray(answer_vectors, dtype=K.floatx())
        
        self.num_samples = question_vectors.shape[0]
        print("Number of samples with embeddings: %d"%self.num_samples)

        print("Loaded the dataset")
        self.dataset = (question_vectors, answer_vectors, all_answers_indices)

    def kg_projection_layer(self, question_vector):
        '''
        Custom layer adding matrix to a tensor
        '''
        kg_projection = K.dot(question_vector, K.constant(self.kg_relation_embeddings_matrix.T))
        return K.dot(kg_projection, K.constant(self.kg_relation_embeddings_matrix))

    def build_model(self):
        '''
        build layers required for training the NN
        '''

        # Q - question embedding input
        question_input = Input(shape=(self.max_question_words, self.word_embs_dim), name='question_input', dtype=K.floatx())
        question_words_embeddings = question_input

        # print K.int_shape(question_words_embeddings)

        # Q' - question encoder
        encoded_question = question_words_embeddings  # model 1 (baseline)

        # encoded_question = EntityLinking(self.kg_word_embeddings_matrix,
        #                                  self.kg_relation_embeddings_matrix,
        #                                  self.word_embs_dim,
        #                                  self.kg_embeddings_dim,
        #                                  self.num_entities)(question_words_embeddings)
        # print K.int_shape(encoded_question)

        # A' - answer decoder
        answer_decoder_output_1 = GRU(self.rnn_units, name='answer_decoder_1', return_sequences=True)(encoded_question)
        answer_decoder_output_2 = GRU(self.rnn_units, name='answer_decoder_2', return_sequences=True)(answer_decoder_output_1)
        answer_decoder_output_3 = GRU(self.rnn_units, name='answer_decoder_3', return_sequences=True)(answer_decoder_output_2)
        answer_decoder_output_4 = GRU(self.rnn_units, name='answer_decoder_4', return_sequences=True)(answer_decoder_output_3)
        # answer_decoder_output_4 = GRU(self.rnn_units, name='answer_decoder_4')(answer_decoder_output_3)

        # A - answer
        answer_output = GRU(self.kg_embeddings_dim, name='answer_decoder')(answer_decoder_output_4)
        # answer_output = Dense(self.num_entities, activation='softmax', name='answer_output')(answer_decoder_output_4)
        
        # K - KG projection
        # kg_projection = Lambda(self.kg_projection_layer, name='answer_selection')(question_encoder_output)  # model 3

        # answer_output = kg_projection

        self.model_train = Model(inputs=[question_input],   # input question
                                 outputs=[answer_output])  # ground-truth target answer set
        print(self.model_train.summary())

    def train(self, batch_size, epochs, lr):
        # define loss
       
        if self.output_vector == 'embedding':
            self.model_train.compile(optimizer='rmsprop', loss='cosine_proximity')
            # self.model_train.compile(optimizer=Adam(lr=lr), loss='cosine_proximity')
            # self.model_train.compile(optimizer=Adadelta(lr=1), loss='cosine_proximity')

        if self.output_vector == 'one-hot':
            # self.model_train.compile(optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
            #                          loss='categorical_crossentropy')
            self.model_train.compile(optimizer='rmsprop', loss='categorical_crossentropy')
            # self.model_train.compile(optimizer=Adadelta(lr=1), loss='categorical_crossentropy')
            # self.model_train.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy')
            # self.model_train.compile(optimizer=Nadam(), loss='categorical_crossentropy')

        # define callbacks for early stopping
        checkpoint = ModelCheckpoint(self.model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min') 
        callbacks_list = [checkpoint, early_stop]
        
        question_vectors, answer_vectors, all_answers_indices = self.dataset

        self.model_train.fit([question_vectors], [answer_vectors], epochs=epochs, callbacks=callbacks_list, verbose=2, validation_split=0.3, shuffle='batch', batch_size=batch_size)

    def test(self):
        '''
        '''
        with CustomObjectScope({'EntityLinking': EntityLinking}):

            self.model_train = load_model(self.model_path)
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

            hits = 0

            if self.output_vector == 'embedding':
                # calculate pairwise distances (via cosine similarity)
                similarity_matrix = cosine_similarity(predicted_answers_vectors, self.kg_relation_embeddings_matrix)
                # print np.argmax(similarity_matrix, axis=1)
                n = 5
                # indices of the top n predicted answers for every question in the test set
                top_ns = similarity_matrix.argsort(axis=1)[:, -n:][::-1]
                # print top_ns[:2]

                for i, answers in enumerate(all_answers_indices):
                    # check if the correct and predicted answer sets intersect
                    correct_answers = set.intersection(set(answers), set(top_ns[i]))
                    if correct_answers:
                        # print correct_answers
                        hits += 1
                print("Hits in top %d: %d/%d"%(n, hits, len(all_answers_indices)))

            elif self.output_vector == 'one-hot':
                all_predicted_answers = Counter()
                for i, answers in enumerate(all_answers_indices):
                    predicted_answer_index = np.argmax(predicted_answers_vectors[i])
                    all_predicted_answers[predicted_answer_index] += 1
                    if predicted_answer_index in answers:
                        hits += 1
                print("Correct answers: %d/%d"%(hits, len(all_answers_indices)))
                print all_predicted_answers


def main(mode):
    '''
    Train model by running: python kbqa_modeli.py train
    '''

    model = KBQA(rnn_units, output_vector)

    # mode switch
    if 'train' in mode.split('/'):
        model.load_data(dataset_name, 'train')

        # build model
        model.build_model()
        # train model
        model.train(batch_size, epochs, lr=learning_rate)
    
    if 'test' in mode.split('/'):
        # use train data
        if not model.dataset:
            model.load_data('train', max_question_words)
        model.test()


if __name__ == '__main__':
    set_random_seed()
    main(sys.argv[1])
