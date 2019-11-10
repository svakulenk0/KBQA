#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Nov 8, 2019

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

QA request-handling functions
'''
import pickle as pkl
from collections import defaultdict
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize, binarize

from keras.preprocessing.text import text_to_word_sequence

from hdt import HDTDocument

from setup import *
from models import *

# path to KG relations
hdt_path = '/mnt/ssd/sv/'
hdt_file = 'dbpedia2016-04en.hdt'
namespace = 'predef-dbpedia2016-04'

model_path = '/home/zola/Projects/temp/KBQA/src/'
embeddings_choice='glove840B300d'
question_types = ['SELECT', 'ASK', 'COUNT']

qt_model = None
ep_model = None


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
        with open(model_path+'qtype_lcquad_%s.pkl'%(embeddings_choice), 'rb') as f:
            self.model_settings = pkl.load(f)
        
        global qt_model
        global ep_model
        
        qt_model = build_qt_inference_model(self.model_settings)
        qt_model.load_weights(model_path+'checkpoints/_qtype_weights.best.hdf5', by_name=True)

        # load pre-trained question parsing model
        with open(model_path+'lcquad_%s.pkl'%(embeddings_choice), 'rb') as f:
            ep_model_settings = pkl.load(f)
        ep_model = build_ep_inference_model(ep_model_settings)
        # load weights
        # ep_model.load_weights('checkpoints/_'+modelname+'_weights.best.hdf5', by_name=True)
        ep_model.load_weights(model_path+'model/2hops-types.h5', by_name=True)

        # connect to the knowledge graph hdt file
        self.kg = HDTDocument(hdt_path+hdt_file)

    # functions for entity linking and relation detection
    def entity_linking(self, e_spans, verbose=False, cutoff=500, threshold=0): 
        guessed_ids = []
        for span in e_spans:
            span_ids = self.e_index.label_scores(span, top=cutoff, threshold=threshold, verbose=verbose, scale=0.3, max_degree=50000)
            guessed_ids.append(span_ids)
        return guessed_ids

    def relation_detection(self, p_spans, verbose=False, cutoff=500, threshold=0.0): 
        guessed_ids = []
        for span in p_spans:
            span_ids = {}
            guessed_labels = []
            if span in self.p_vectors:
                guessed_labels.append([span, 1])
            for p, score in self.p_vectors.most_similar(span, topn=cutoff):
                if score >= threshold:
                    guessed_labels.append([p, score])
            for label, score in guessed_labels:
                for match in self.p_index.look_up_by_label(label):
                    _id = match['_source']['id']
                    span_ids[_id] = score
                    if verbose:
                        uri = match['_source']['uri']
                        print(uri)
                        print(score)
            guessed_ids.append(span_ids)
        return guessed_ids

    # MP functions
    def generate_adj_sp(self, adjacencies, n_entities, include_inverse):
        '''
        Build adjacency matrix
        '''
        adj_shape = (n_entities, n_entities)
        # colect all predicate matrices separately into a list
        sp_adjacencies = []

        for edges in adjacencies:
            # split subject (row) and object (col) node URIs
            n_edges = len(edges)
            row, col = np.transpose(edges)
            
            # duplicate edges in the opposite direction
            if include_inverse:
                _row = np.hstack([row, col])
                col = np.hstack([col, row])
                row = _row
                n_edges *= 2
            
            # create adjacency matrix for this predicate
            data = np.ones(n_edges)
            adj = sp.csr_matrix((data, (row, col)), shape=adj_shape)
            sp_adjacencies.append(adj)
        
        return np.asarray(sp_adjacencies)

    def hop(self, entities, constraints, top_predicates, verbose=False, max_triples=500000, bl_p=[68655]):
        '''
        Extract the subgraph for the selected entities
        bl_p  -- the list of predicates to ignore (e.g. type predicate is too expensive to expand)
        ''' 
    #     print(top_predicates)
        n_constraints = len(constraints)
        if entities:
            n_constraints += 1

        top_entities = entities + constraints
        all_entities_ids = [_id for e in top_entities for _id in e]
        top_predicates_ids = [_id for p in top_predicates for _id in p if _id not in bl_p]

        # iteratively call the HDT API to retrieve all subgraph partitions
        activations = defaultdict(int)
        offset = 0

        while True:
            # get the subgraph for selected predicates only
    #         print(top_predicates_ids)
            self.kg.configure_hops(1, top_predicates_ids, namespace, True)
            entities, predicate_ids, adjacencies = self.kg.compute_hops(all_entities_ids, max_triples, offset)
    #         print(adjacencies)
            # show subgraph entities
    #         print([e_index.look_up_by_id(e)[0]['_source']['uri'] for e in entities])
            
            if not entities:
                answers = [{a_id: a_score} for a_id, a_score in activations.items()]
                return answers

            if verbose:
                print("Subgraph extracted:")
                print("%d entities"%len(entities))
                print("%d predicates"%len(predicate_ids))
                print("Loading adjacencies..")

            offset += max_triples
            # index entity ids global -> local
            entities_dict = {k: v for v, k in enumerate(entities)}
            # generate a list of adjacency matrices per predicate assuming the graph is undirected wo self-loops
            A = self.generate_adj_sp(adjacencies, len(entities), include_inverse=True)
    #         print(predicate_ids)
            # activate entities -- build sparse matrix
            row, col, data = [], [], []
            for i, concept_ids in enumerate(top_entities):
                for entity_id, score in concept_ids.items():
                    if entity_id in entities_dict:
    #                     print(e_index.look_up_by_id(entity_id)[0]['_source']['uri'])
    #                     print(score)
                        local_id = entities_dict[entity_id]
                        row.append(i)
                        col.append(local_id)
                        data.append(score)
            x = sp.csr_matrix((data, (row, col)), shape=(len(top_entities), len(entities)))
        
            # iterate over predicates
            ye = sp.csr_matrix((len(top_entities), len(entities)))
            # activate predicates
            if top_predicates_ids:
                yp = sp.csr_matrix((len(top_predicates), len(entities)))
                for i, concept_ids in enumerate(top_predicates):
                    # activate predicates
                    p = np.zeros([len(predicate_ids)])
                    # iterate over synonyms
                    for p_id, score in concept_ids.items():
                        if p_id in predicate_ids:
                            local_id = predicate_ids.index(p_id)
                            p[local_id] = score
                    # slice A by the selected predicates
                    _A = sum(p*A)
                    _y = x @ _A
                    # normalize: cut top to 1
                    _y[_y > 1] = 1
                    yp[i] = _y.sum(0)
                    ye += _y
                y = sp.vstack([ye,yp])
            # fall back to evaluate all predicates
            else:
                y = x @ sum(A)
            sum_a = sum(y)
            sum_a_norm = sum_a.toarray()[0] / (len(top_predicates) + n_constraints) #normalize(sum_a, norm='max', axis=1).toarray()[0]
            # normalize: cut top to 1
            sum_a_norm[sum_a_norm > 1] = 1
            # activations across components
            y_counts = binarize(y, threshold=0.0)
            count_a = sum(y_counts).toarray()[0]
            # final scores
            y = (sum_a_norm + count_a) / (len(top_predicates) + n_constraints + 1)

            # check output size
            assert y.shape[0] == len(entities)

            top = np.argwhere(y > 0).T.tolist()[0]
            if len(top) > 0:
                activations1 = np.asarray(entities)[top]
                # store the activation values per id answer id
                for i, e in enumerate(entities):
                    if e in activations1:
                        activations[e] += y[i]
            # if not such answer found fall back to return the answers satisfying max of the constraints
            else:
                # select answers that satisfy maximum number of constraints
                y_p = np.argmax(y)
                # maximum number of satisfied constraints
                max_cs = y[y_p]
                # at least some activation (evidence from min one constraint)
                if max_cs != 0:
                    # select answers
                    top = np.argwhere(y == max_cs).T.tolist()[0]
                    activations1 = np.asarray(entities)[top]
                    # store the activation values per id answer id
                    for i, e in enumerate(entities):
                        if e in activations1:
                            activations[e] += y[i]

    def request(self, question, top_n=1, verbose=False):
        # parse question into words and embed
        x_test_sent = np.zeros((self.model_settings['max_len'], self.model_settings['emb_dim']))
        q_words = text_to_word_sequence(question)
        for i, word in enumerate(q_words):
            x_test_sent[i] = self.word_vectors.query(word)

        # predict question type
        if verbose:
            print(x_test_sent)
        y_p = qt_model.predict(np.array([x_test_sent]))
        y_p = np.argmax(y_p, axis=-1)[0]
        p_qt = question_types[y_p]
        ask_question = p_qt == 'ASK'
        print(p_qt)

        # use GS spans + preprocess
        y_p = ep_model.predict(np.array([x_test_sent]))
        y_p = np.argmax(y_p, axis=-1)[0]
        e_spans1 = collect_mentions(q_words, y_p, 1)
        p_spans1 = collect_mentions(q_words, y_p, 2)
        p_spans2 = collect_mentions(q_words, y_p, 3)

        #         c_spans1 = doc['c1_spans']
        #         c_spans2 = doc['c2_spans']

        # match predicates
        top_predicates_ids1 = self.relation_detection(p_spans1, threshold=0)
        top_predicates_ids2 = self.relation_detection(p_spans2, threshold=0)

        # use GS classes
        #         classes1 = [{_id: 1} for _id in doc['classes_ids'] if _id in doc['1hop_ids'][0]]
        #         classes2 = [{_id: 1} for _id in doc['classes_ids'] if _id in doc['2hop_ids'][0]]

        top_entities_ids1 = self.entity_linking(e_spans1, threshold=0.7)

        if ask_question:
            a_threshold = 0.0
        else:
            a_threshold = 0.5

        # MP
        answers_ids = []

        # 1st hop
        answers_ids1 = self.hop([], top_entities_ids1, top_predicates_ids1, verbose)
        #         if classes1:
        #             answers_ids1 = filter_answer_by_class(classes1, answers_ids1)
        answers1 = [{a_id: a_score} for activations in answers_ids1 for a_id, a_score in activations.items() if a_score > a_threshold]

        # 2nd hop
        if top_predicates_ids1 and top_predicates_ids2:                
            answers_ids = self.hop(answers1, [], top_predicates_ids2, verbose)
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
        print([{self.e_index.look_up_by_id(_id)[0]['_source']['uri']: score} for answer in top_entities_ids1 for _id, score in answer.items() if self.e_index.look_up_by_id(_id)][:top_n])
        print([{self.p_index.look_up_by_id(_id)[0]['_source']['uri']: score} for answer in top_predicates_ids1 for _id, score in answer.items() if self.p_index.look_up_by_id(_id)][:top_n])
        print([{self.p_index.look_up_by_id(_id)[0]['_source']['uri']: score} for answer in top_predicates_ids2 for _id, score in answer.items() if self.p_index.look_up_by_id(_id)][:top_n])


        # show intermediate answers if there was a second hop
        if top_predicates_ids2:
            print([{self.e_index.look_up_by_id(_id)[0]['_source']['uri']: score} for answer in answers1 for _id, score in answer.items() if self.e_index.look_up_by_id(_id)][:top_n])


        if ask_question:
            # make sure the output matches every input basket
            all_entities_baskets = [set(e.keys()) for e in top_entities_ids1]
            answers = all(x & set(answers_ids) for x in all_entities_baskets)
        else:
            # show answers
            answers = [{self.e_index.look_up_by_id(_id)[0]['_source']['uri']: score} for answer in answers for _id, score in answer.items() if self.e_index.look_up_by_id(_id)][:top_n]
        
        if verbose:
            print(answers)

        return answers

    def test_request(self):
        question = "What are some other works of the author of The Phantom of the Opera?"
        self.request(question, verbose=True)


if __name__ == '__main__':
    service = KBQA()
    service.test_request()
