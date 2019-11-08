#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Nov 8, 2019

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

QA request-handling functions
'''
from collections import defaultdict
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize, binarize

from setup import *
import pickle as pkl
from models import build_qt_inference_model, build_ep_inference_model

# path to KG relations
hdt_path = '/mnt/ssd/sv/'
hdt_file = 'dbpedia2016-04en.hdt'
namespace = 'predef-dbpedia2016-04'

model_path = '/home/zola/Projects/temp/KBQA/src/'
embeddings_choice='glove840B300d'
question_types = ['SELECT', 'ASK', 'COUNT']


class KBQA():
    def __init__(self, dataset_name='lcquad'):
        '''
        Setup models, indices, embeddings and connection to the KG through the HDT API
        '''
        
        # connect to the entity and predicate catalogs
        self.e_index = IndexSearch('dbpedia201604e')
        self.p_index = IndexSearch('dbpedia201604p')

        # load embeddings
        self.word_vectors = load_embeddings(embeddings_choice)
        self.p_vectors = load_embeddings('fasttext_p_labels')
        
        # load pre-trained question type classification model
        modelname = 'qtype'
        with open(model_path+'%s_%s_%s.pkl'%(modelname, dataset_name, embeddings_choice), 'rb') as f:
            self.model_settings = pkl.load(f)
        self.qt_model = build_qt_inference_model(self.model_settings)
        self.qt_model.load_weights('checkpoints/_'+modelname+'_weights.best.hdf5', by_name=True)

        # load pre-trained question parsing model
        with open(model_path+'%s_%s.pkl'%(dataset_name, embeddings_choice), 'rb') as f:
            ep_model_settings = pkl.load(f)
        self.ep_model = build_ep_inference_model(ep_model_settings)
        # load weights
        # ep_model.load_weights('checkpoints/_'+modelname+'_weights.best.hdf5', by_name=True)
        self.ep_model.load_weights(model_path+'model/'+modelname+'.h5', by_name=True)

        # connect to the knowledge graph hdt file
        self.kg = HDTDocument(hdt_path+hdt_file)

    def request(self, question, verbose=False):
        # parse question into words and embed
        x_test_sent = np.zeros((self.model_settings['max_len'], self.model_settings['emb_dim']))
        q_words = text_to_word_sequence(q)
        for i, word in enumerate(q_words):
            x_test_sent[i] = self.word_vectors.query(word)

        # predict question type
        y_p = self.qt_model.predict(np.array([x_test_sent]))
        y_p = np.argmax(y_p, axis=-1)[0]
        p_qt = question_types[y_p]
        ask_question = p_qt == 'ASK'
        print(p_qt)

        # use GS spans + preprocess
        y_p = self.ep_model.predict(np.array([x_test_sent]))
        y_p = np.argmax(y_p, axis=-1)[0]
        e_spans1 = collect_mentions(q_words, y_p, 1)
        p_spans1 = collect_mentions(q_words, y_p, 2)
        p_spans2 = collect_mentions(q_words, y_p, 3)

        #         c_spans1 = doc['c1_spans']
        #         c_spans2 = doc['c2_spans']

        # match predicates
        top_predicates_ids1 = relation_detection(p_spans1, threshold=0)
        top_predicates_ids2 = relation_detection(p_spans2, threshold=0)

        # use GS classes
        #         classes1 = [{_id: 1} for _id in doc['classes_ids'] if _id in doc['1hop_ids'][0]]
        #         classes2 = [{_id: 1} for _id in doc['classes_ids'] if _id in doc['2hop_ids'][0]]

        top_entities_ids1 = entity_linking(e_spans1, threshold=0.7)

        if ask_question:
            a_threshold = 0.0
        else:
            a_threshold = 0.5

        # MP
        answers_ids = []

        # 1st hop
        answers_ids1 = hop([], top_entities_ids1, top_predicates_ids1, verbose)
        #         if classes1:
        #             answers_ids1 = filter_answer_by_class(classes1, answers_ids1)
        answers1 = [{a_id: a_score} for activations in answers_ids1 for a_id, a_score in activations.items() if a_score > a_threshold]

        # 2nd hop
        if top_predicates_ids1 and top_predicates_ids2:                
            answers_ids = hop(answers1, [], top_predicates_ids2, verbose)
        #             if classes2:
        #                 answers_ids = filter_answer_by_class(classes2, answers_ids)
            answers = [{a_id: a_score} for activations in answers_ids for a_id, a_score in activations.items() if a_score > a_threshold]
        else:
            answers = answers1

        answers_ids = [_id for a in answers for _id in a]


        # show spans
        print(e_spans1)
        print(p_spans1)
        print(p_spans2)

        # show  matches
        top_n = 1
        print([{self.e_index.look_up_by_id(_id)[0]['_source']['uri']: score} for answer in top_entities_ids1 for _id, score in answer.items() if self.e_index.look_up_by_id(_id)][:top_n])
        print([{self.p_index.look_up_by_id(_id)[0]['_source']['uri']: score} for answer in top_predicates_ids1 for _id, score in answer.items() if self.p_index.look_up_by_id(_id)][:top_n])
        print([{self.p_index.look_up_by_id(_id)[0]['_source']['uri']: score} for answer in top_predicates_ids2 for _id, score in answer.items() if self.p_index.look_up_by_id(_id)][:top_n])


        # show intermediate answers if there was a second hop
        if top_predicates_ids2:
            print([{self.e_index.look_up_by_id(_id)[0]['_source']['uri']: score} for answer in answers1 for _id, score in answer.items() if self.e_index.look_up_by_id(_id)][:top_n])


        if ask_question:
            # make sure the output matches every input basket
            all_entities_baskets = [set(e.keys()) for e in top_entities_ids1]
            answer = all(x & set(answers_ids) for x in all_entities_baskets)
            print(answer)
        else:
            # show answers
            print([{self.e_index.look_up_by_id(_id)[0]['_source']['uri']: score} for answer in answers for _id, score in answer.items() if self.e_index.look_up_by_id(_id)][:top_n])


    def test_request(self):
        question = "What are some other works of the author of The Phantom of the Opera?"
        self.request(question, verbose=True)


if __name__ == '__main__':
    service = KBQA()
    service.test_request()
