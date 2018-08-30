from __future__ import print_function

from rgcn_data_preprocess import load_data

from rgcn_utils import sp

import pickle as pkl

import os
import sys
import time
import argparse

# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", type=str, default="aifb",
#                 help="Dataset string ('aifb', 'mutag', 'bgs', 'am')")

# args = vars(ap.parse_args())

# print(args)

# # Define parameters
# DATASET = args['dataset']


import sys
rdfFileName = sys.argv[1]
working_dir = sys.argv[2]
dataset_str = sys.argv[3]

# Get data
A, X, rel_dict = load_data(rdfFileName, working_dir, dataset_str)

rel_list = range(len(A))
for key, value in rel_dict.iteritems():
    if value * 2 >= len(A):
        continue
    rel_list[value * 2] = key
    rel_list[value * 2 + 1] = key + '_INV'


num_nodes = A[0].shape[0]
A.append(sp.identity(A[0].shape[0]).tocsr())  # add identity matrix

support = len(A)

print("Relations used and their frequencies" + str([a.sum() for a in A]))

## MC: removed because this seems to be an optmimization only valid for label prediction
# print("Calculating level sets...")
# t = time.time()
# # Get level sets (used for memory optimization)
# bfs_generator = bfs_relational(A, labeled_nodes_idx)
# lvls = list()
# lvls.append(set(labeled_nodes_idx))
# lvls.append(set.union(*bfs_generator.next()))
# print("Done! Elapsed time " + str(time.time() - t))

# # Delete unnecessary rows in adjacencies for memory efficiency
# todel = list(set(range(num_nodes)) - set.union(lvls[0], lvls[1]))
# for i in range(len(A)):
#     csr_zero_rows(A[i], todel)

data = {'A': A, }

dirname = os.path.dirname(os.path.realpath(working_dir))

with open(dirname + '/' + dataset_str + '.pickle', 'wb') as f:
    pkl.dump(data, f, pkl.HIGHEST_PROTOCOL)
