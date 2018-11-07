#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Nov 7, 2018

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Neural network model with optimal attention priors for complex KBQA

'''
import tensorflow as tf

from utils import *


class KBQA:
    '''
    Neural network model with optimal attention priors for complex KBQA
    '''

    def __init__(self, output_model_path='./models/model.best.hdf5'):
        '''
        output_model_path - path to store the pre-trained model
        '''
        self.output_model_path = output_model_path
        self.dataset = None

    def load_data(self, dataset_name, split):
        '''
        '''
        pass

    def build_model(self):
        '''
        Model architecture
        '''
        # define layers and connections between them 
        # TODO

        # define input/output
        self.model_train = Model(inputs=[question_input],   # input question
                                 outputs=[answer_output])  # ground-truth target answer
        # show model architecture
        print(self.model_train.summary())

    def train(self, batch_size, epochs, validation_split, lr):
        '''
        '''
        # define callbacks for early stopping
        checkpoint = ModelCheckpoint(self.model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min') 
        callbacks_list = [checkpoint, early_stop]
        
        # load data
        # TODO

        # start training
        self.model_train.fit([question_vectors], [answer_vectors], epochs=epochs,
                             callbacks=callbacks_list, verbose=2, validation_split=0.3,
                             shuffle='batch', batch_size=batch_size)

    def test(self):
        '''
        '''
        self.model_train = load_model(self.model_path)
        print("Loaded the pre-trained model")

        # load data
        # TODO
        
        # test
        predicted_answers_vectors = self.model_train.predict(question_vectors)

        # compare predicted answers with correct answers
        # TODO


def main(mode):
    '''
    Train model by running: python kbqa_modeli.py train
    python kbqa_modeli.py test to only test the model
    or python kbqa_modeli.py train/test for both
    '''
    # define all the model parameters
    dataset_name = 'lcquad'

    # model architecture
    max_question_words = 24  # input layer size

    # model training
    batch_size = 32
    epochs = 50  # 10
    learning_rate = 0.001
    validation_split = 0.3

    # make sure GPU is enabled
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    model = KBQA()

    # mode switch
    if 'train' in mode.split('/'):
        # build model
        model.build_model()
        # train model
        model.train(batch_size, epochs, validation_split, learning_rate)
    if 'test' in mode.split('/'):
        model.test()


if __name__ == '__main__':
    set_random_seed()
    main(sys.argv[1])
