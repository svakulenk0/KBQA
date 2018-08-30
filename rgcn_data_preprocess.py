'''
Created on Aug 28, 2018

.. codeauthor: Michael Cochez

Preprocessing for rGCN

Based on RGCN https://github.com/tkipf/relational-gcn
    Specifically https://github.com/tkipf/relational-gcn/blob/master/rgcn/data_utils.py
    Changes are in load_data. We do use it for loading labels as our set-up is more complex.
    We only need to get the adjecancy matrix for the layers.
'''

from __future__ import print_function

import os
import re
import sys
import gzip
import numpy as np
import scipy.sparse as sp
import rdflib as rdf
import glob
import pandas as pd
import wget
import pickle as pkl

from collections import Counter

np.random.seed(123)


class RDFReader:
    __graph = None
    __freq = {}

    def __init__(self, file):

        self.__graph = rdf.Graph()

        if file.endswith('nt.gz'):
            with gzip.open(file, 'rb') as f:
                self.__graph.parse(file=f, format='nt')
        else:
            self.__graph.parse(file, format=rdf.util.guess_format(file))

        # See http://rdflib.readthedocs.io for the rdflib documentation

        self.__freq = Counter(self.__graph.predicates())

        print("Graph loaded, frequencies counted.")

    def triples(self, relation=None):
        for s, p, o in self.__graph.triples((None, relation, None)):
            yield s, p, o

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__graph.destroy("store")
        self.__graph.close(True)

    def subjectSet(self):
        return set(self.__graph.subjects())

    def objectSet(self):
        return set(self.__graph.objects())

    def relationList(self):
        """
        Returns a list of relations, ordered descending by frequenecy
        :return:
        """
        res = list(set(self.__graph.predicates()))
        res.sort(key=lambda rel: - self.freq(rel))
        return res

    def __len__(self):
        return len(self.__graph)

    def freq(self, relation):
        """
        The frequency of this relation (how many distinct triples does it occur in?)
        :param relation:
        :return:
        """
        if relation not in self.__freq:
            return 0
        return self.__freq[relation]


def load_sparse_csr(filename):
    loader = np.load(filename)
    return sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape=loader['shape'], dtype=np.float32)


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_data(graph_file, working_dir, dataset_str, limit=-1):
    """

    :param graph_file str: the path to the rdf serialization of the graph
    :param working_dir str: the directory in which to store the preprocessed data
    :param dataset_str str: the name to be used in the paths for this dataset
    :param limit: If > 0, will only load this many adj. matrices
        All adjacencies are preloaded and saved to disk,
        but only a limited a then restored to memory.
    :return:
    """
    import errno    
    import os

    def mkdir_p(path):
        try:
            os.makedirs(path)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                raise

    def mkdir_p_for_file(thefile):
        mkdir_p(os.path.dirname(thefile))
    
    assert limit == -1 or limit > 0

    print('Loading dataset', graph_file)

    adj_fprepend = 'data/' + dataset_str + '/adjacencies_'
    rel_dict_file = 'data/' + dataset_str + '/rel_dict.pkl'
    nodes_file = 'data/' + dataset_str + '/nodes.pkl'
    #The nodes_file contains the node labels as instances of rdflib.term.URIRef, this stores it as just strings.
    #MC: I am not sure whether both are really needed. 
    nodes_strings_file = 'data/' + dataset_str + '/nodes_strings.pkl'

    dirname = os.path.dirname(working_dir)
    adj_fprepend = dirname + '/' + adj_fprepend
    rel_dict_file = dirname + '/' + rel_dict_file
    nodes_file = dirname + '/' + nodes_file
    nodes_strings_file = dirname + '/' + nodes_strings_file

    adj_files = glob.glob(adj_fprepend + '*.npz')

    mkdir_p(dirname)
    mkdir_p_for_file(adj_fprepend)
    mkdir_p_for_file(rel_dict_file)

    if adj_files != []:

        # load precomputed adjacency matrix

        adj_files.sort(key=lambda f: int(re.search('adjacencies_(.+?).npz', f).group(1)))

        if limit > 0:
            adj_files = adj_files[:limit * 2]

        adjacencies = [load_sparse_csr(file) for file in adj_files]
        adj_shape = adjacencies[0].shape

        print('Number of nodes: ', adj_shape[0])
        print('Number of relations: ', len(adjacencies))

        relations_dict = pkl.load(open(rel_dict_file, 'rb'))

    else:

        with RDFReader(graph_file) as reader:

            relations = reader.relationList()
            subjects = reader.subjectSet()
            objects = reader.objectSet()

            print([(rel, reader.freq(rel)) for rel in relations[:limit]])

            nodes = list(subjects.union(objects))
            adj_shape = (len(nodes), len(nodes))

            print('Number of nodes: ', len(nodes))
            print('Number of relations in the data: ', len(relations))

            relations_dict = {rel: i for i, rel in enumerate(list(relations))}
            nodes_dict = {node: i for i, node in enumerate(nodes)}

            assert len(nodes_dict) < np.iinfo(np.int32).max

            adjacencies = []

            for i, rel in enumerate(
                    relations if limit < 0 else relations[:limit]):

                print(
                    u'Creating adjacency matrix for relation {}: {}, frequency {}'.format(
                        i, rel, reader.freq(rel)))
                edges = np.empty((reader.freq(rel), 2), dtype=np.int32)

                size = 0
                for j, (s, p, o) in enumerate(reader.triples(relation=rel)):
                    if nodes_dict[s] > len(nodes) or nodes_dict[o] > len(nodes):
                        print(s, o, nodes_dict[s], nodes_dict[o])

                    edges[j] = np.array([nodes_dict[s], nodes_dict[o]])
                    size += 1

                print('{} edges added'.format(size))

                row, col = np.transpose(edges)

                data = np.ones(len(row), dtype=np.int8)

                adj = sp.csr_matrix((data, (row, col)), shape=adj_shape,
                                    dtype=np.int8)

                adj_transp = sp.csr_matrix((data, (col, row)), shape=adj_shape,
                                           dtype=np.int8)

                save_sparse_csr(adj_fprepend + '%d.npz' % (i * 2), adj)
                save_sparse_csr(adj_fprepend + '%d.npz' % (i * 2 + 1),
                                adj_transp)

                if limit < 0:
                    adjacencies.append(adj)
                    adjacencies.append(adj_transp)

        # Reload the adjacency matrices from disk
        # if limit < 0, then they are already/still loaded 
        if limit > 0:
            adj_files = glob.glob(adj_fprepend + '*.npz')
            adj_files.sort(key=lambda f: int(
                re.search('adjacencies_(.+?).npz', f).group(1)))

            adj_files = adj_files[:limit * 2]
            for i, file in enumerate(adj_files):
                adjacencies.append(load_sparse_csr(file))
                print('%d adjacency matrices loaded ' % i)

        pkl.dump(nodes, open(nodes_file, 'wb'))
        stringNodes = [str(node) for node in nodes]
        pkl.dump(stringNodes, open(nodes_strings_file, 'wb'))
        
        pkl.dump(relations_dict, open(rel_dict_file, 'wb'))


    # this loads the initial features as 1 hot encoded vectors.
    features = sp.identity(adj_shape[0], format='csr')

    return adjacencies, features, relations_dict


def parse(symbol):
    if symbol.startswith('<'):
        return symbol[1:-1]
    return symbol


def to_unicode(input):
    if isinstance(input, unicode):
        return input
    elif isinstance(input, str):
        return input.decode('utf-8', errors='replace')
    return str(input).decode('utf-8', errors='replace')



if __name__ == "__main__":
    import sys
    rdfFileName = sys.argv[1]
    working_dir = sys.argv[2]
    dataset_str = sys.argv[3]
    load_data(rdfFileName, working_dir, dataset_str)





