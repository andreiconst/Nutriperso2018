#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 14:55:47 2018

@author: andrei
"""

import numpy as np
import logging
import pandas as pd
import sys
sys.path.insert(0, '/home/andrei/Desktop/kantar_2014/code_produits/')


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    filename='/tmp/myapp.log',
                    filemode='w')


from scipy.sparse import csr_matrix, tril

from tools_preprocessing import *
from glove_tf import *
from tools_clustering import *


logger = logging.getLogger("glove")

purchase_matrix, row_names, col_names = load_purchase_matrix('data_cleaned/purchase_table_codepanier.npz', 
                                                             'data_cleaned/households_codepanier.txt', 
                                                             'data_cleaned/products_matrix_full.txt')


products_table = pd.read_csv('data_cleaned/produits_achats.csv', encoding='latin1')
products_table = clean_table(products_table)


cluster_table = pd.read_csv('data_cleaned/cluster_products_mano.csv', encoding='latin1')

purchase_matrix_clustered,  col_names = cluster_products(cluster_table, products_table, purchase_matrix, col_names, verbose = False, )

purchase_matrix_clustered = purchase_matrix_clustered.tocoo()


##############################################################################

#
#weird_indexes = np.where(purchase_matrix_clustered.data > 200)[0]
#
#for i in weird_indexes:
#    if purchase_matrix_clustered.col[i] != 14848:
#        print(purchase_matrix_clustered.col[i])
#
#col_names[3636]
#prod_weird = products_table.loc[products_table['product'] == 115687]


#Attention weird
# [71960, 286729, ] weird products please do something asap

##############################################################################




purchase_matrix_trimed, col_names = trim_sparse_columns(purchase_matrix_clustered, col_names, 10)
purchase_matrix_trimed = csr_matrix((np.ones(len(purchase_matrix_trimed.data)), purchase_matrix_trimed.indices, purchase_matrix_trimed.indptr)).tocsc()

co_occurences = np.dot(purchase_matrix_trimed, purchase_matrix_trimed.T).tocsr()
csr_setdiag_val(co_occurences)

model = GloVeModel(col_names, 100, batch_size=2048, cooccurrence_cap = 100, learning_rate= 0.05)
model.train(30, co_occurences.tocoo(), log_dir='tf_glove')



embeddings = model.embeddings /2

#
#index = np.asarray(range(10000))
#shuffle(index)
#index_list = list()
#
#for i in range(int(10000/1000)):
#    index_list.append([i + j * int(10000/1000) for j in range(1000)])
# 
#result = list()
#for l in index_list:
#    for k in l:
#        result.append(k)
#
#sorted(result)


def check_error(model):
    embeddings = model.embeddings
    np.random.seed(123)
    indexes = np.random.randint(0, len(co_occurences.data), 10000)
    errors = np.zeros(10000)
    
    
    for i,j in enumerate(indexes):
        errors[i] = np.abs(np.log(1+co_occurences.data[j]) - (np.dot(embeddings[co_occurences.row[j],:].T, embeddings[co_occurences.col[j],:]))) 

    return errors

co_occurences = co_occurences.tocoo()
error = check_error(model)
np.mean(error)


embeddings_play = pd.DataFrame(embeddings, index = col_names)
embeddings_play.to_csv('data_cleaned/embeddings_products_cp_mano100.csv')