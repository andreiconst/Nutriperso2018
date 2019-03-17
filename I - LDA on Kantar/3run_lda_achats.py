#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 16:36:54 2018

@author: andrei
"""

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import csv
from scipy.sparse import csr_matrix
import pandas as pd


import sys
import os
sys.path.insert(0, os.getcwd() + '/Tools/')

from draw_graphs_households import *
from tools_regression import *
from tools_clustering import *
from tools_preprocessing import *

households = import_and_transform_households('data/foyers_traites.csv')
#week_table = import_week_table('data/household_activity_week.csv')
#circuit_table = import_circuit_table('data/household_activity_circuit.csv')

#households = merge_households(households, week_table,  circuit_table)

purchase_matrix, row_names, col_names = load_purchase_matrix('data/purchase_table_full.npz', 
                                                             'data/households_matrix_full.txt', 
                                                             'data/products_matrix_full.txt')

products_table = pd.read_csv('data/produits_achats.csv', encoding='latin1')
products_table = clean_table(products_table)
cluster_table = pd.read_csv('data/cluster_products_auto.csv', encoding='latin1')
purchase_matrix,  col_names = cluster_products(cluster_table, products_table, purchase_matrix, col_names, verbose = True, )
#purchase_matrix = csr_matrix((np.ones(len(purchase_matrix.data)), purchase_matrix.indices, purchase_matrix.indptr)).tocsc()

purchase_matrix, col_names = trim_sparse_columns(purchase_matrix, col_names, 10)
#purchase_matrix, row_names = delete_max_rows(purchase_matrix, row_names, 70)
purchase_matrix, row_names = delete_rows_entropy(purchase_matrix, row_names, 1.5)



indexes = np.random.rand(purchase_matrix.shape[0]) < 0.75
train = purchase_matrix[indexes, :]
validation = purchase_matrix[~indexes, :]


list_perplexity_validation = list()
#achats_households_final = normalise_products_purchase(achats_households_normalised)
for topics in [[10,10],[20,10],[30,15],[50,15]]:
        
    model = LatentDirichletAllocation(n_topics=topics[0], random_state=123, n_jobs = 1, max_iter = topics[1])
    id_topic = model.fit_transform(train)
    list_perplexity_validation.append([topics[0], model.perplexity(validation)])

model = LatentDirichletAllocation(n_topics=30, random_state=123, n_jobs = 1, max_iter = 15)
id_topic = model.fit_transform(purchase_matrix)    

documents = pd.DataFrame(id_topic, index = row_names)
documents.to_csv('data/documents_lda_auto.csv', index=True )

topics = model.components_.T
topics = pd.DataFrame(topics, index = col_names)
topics.to_csv('data/topics_lda_auto.csv', index = True)








#def load_sparse_csr(filename):
#    loader = np.load(filename)
#    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
#                      shape=loader['shape'])
#
#
#def create_simple_mapping(mapping_list):
#    result = dict()
#    for i, value in enumerate(mapping_list):
#        result[value] = i
#    return result
#
#def import_products_clustering(name_products):
#    liste_produits = list()
#    with open('data/' + name_products, 'r') as csvfile:
#        reader = csv.reader(csvfile, delimiter=',')
#        for row in reader:
#            
#             liste_produits.append(row[0])
#    liste_produits = [int(p) for p in liste_produits][1:]
#    return liste_produits
#
#
#achats_households = load_sparse_csr('data/purchase_table_mano.npz')
##achats_households = achats_households[:100]
##achats_households = achats_households.toarray()
##description_products = get_description_products(products_clusters,dict_description,products_table)
##description_full = [str(list_products[i]) + ' | ' + description_products[i] for i in range(len(description_products))]
##
##achats_households = pd.DataFrame(achats_households, columns = description_full).T
##achats_households.to_csv('pruchase_matrix_visualisation.csv')
#
#values = achats_households.data
#values = np.log(1+values)
#rows = achats_households.indices
#columns =  achats_households.indptr
#achats_households = csr_matrix((values, rows, columns))
#np.max(achats_households)
#
#list_products = sorted(import_products_clustering('cluster_products_mano.csv'))
#list_households, households = import_households()
#dict_households = create_simple_mapping(list_households)
#
#def entropy(data):
##    assert np.sum(data) == 1
#    assert data.all() >=0
#    data = data[data >0]
#    entropy = -np.dot(data, np.log(data))
#    return entropy
#
#index_list = list()
#for i in range(achats_households.shape[0]):
#    features = achats_households[i,:]
#    data = features.data / np.sum(features.data)
#    if entropy(data) > 1:
#        index_list.append(i)
#
#achats_households = achats_households[index_list,:]
#list_households = [list_households[i] for i in index_list]
#
#assert achats_households.shape[0] == len(list_households)
#assert achats_households.shape[1] == len(list_products)
#

'''
households['n_enf'] = households['en3'] + households['en6'] + households['en15'] + households['en25']
households_trimed = households[['household', 'sexe', 'age','n_enf']]
households_m = households_trimed.loc[households_trimed['sexe'] == 0]
households_f = households_trimed.loc[households_trimed['sexe'] == 1]
household_plane = pd.merge(households_f, households_m, left_on = 'household', right_on = 'household', how = 'outer')
household_plane = household_plane.drop(columns=['n_enf_x'])
household_plane['household'] = household_plane['household'].astype('str')

indexes = np.random.rand(achats_households.shape[0]) < 0.75
train = achats_households[indexes, :]
validation = achats_households[~indexes, :]

list_perplexity_validation = list()
#achats_households_final = normalise_products_purchase(achats_households_normalised)
for topics in [[10,10],[20,10],[30,15],[50,20]]:
        
    model = LatentDirichletAllocation(n_topics=topics[0], random_state=123, n_jobs = 1, max_iter = topics[1])
    id_topic = model.fit_transform(train)
    list_perplexity_validation.append([topics[0], model.perplexity(validation)])

model = LatentDirichletAllocation(n_topics=30, random_state=123, n_jobs = 1, max_iter = 20)
id_topic = model.fit_transform(achats_households)    

documents = pd.DataFrame(id_topic, index = list_households)
documents_final = pd.merge(household_plane, documents, right_index = True, left_on = 'household')
documents_final.to_csv('data/documents_lda_mano.csv', index=False, )

topics = model.components_.T
topics = pd.DataFrame(topics, index = list_products)
topics.to_csv('data/topics_lda_mano.csv', index = True)


'''








'''

def normalize_size_households(matrix, row_labels):
    
    dict_ucfo_val = dict()

    household_size = list()
    households_original = pd.read_csv('data/menages_2014.csv', sep=';', encoding='latin1')
    households_original['household'] = households_original['household'].astype(str)
    
    for i,row in households_original.iterrows():
        dict_ucfo_val[row['household']] = row['ucfo_val'] 

    for row_label in row_labels:
        try:
            household_size.append(dict_ucfo_val[row_label])
        except:
            household_size.append(-1)
            
        
    household_size_vector = np.asarray(household_size)
    
    matrix = matrix.T / household_size_vector
    return matrix.T



def pad(vector):
    indexes = np.where(vector<1e-2)
    
    for i in indexes:
        vector[i] =1
    
    return vector
  
def normalise_products_purchase(matrix): 

    indexes = matrix.nonzero()
    dict_indexes = dict()
    
    for i in range(len(indexes[0])):
        
        try:
            dict_indexes[indexes[1][i]].append(indexes[0][i])
        except:
            dict_indexes[indexes[1][i]] = list()
            dict_indexes[indexes[1][i]].append(indexes[0][i])
            
    means = list()
    stds = list()
    
    for i in range(len(dict_indexes)):
        
        std = np.std(matrix[dict_indexes[i],i].toarray())
        mean = np.mean(matrix[dict_indexes[i],i].toarray())
        
        stds.extend([1.0 / std if std>1e-2 else 1 for j in range(len(dict_indexes[i]))])
        means.extend([mean for j in range(len(dict_indexes[i]))])
    
    matrix_mean = csr_matrix((means, (indexes[0], indexes[1])))
    matrix_stds = csr_matrix((stds, (indexes[0], indexes[1])))
    
    matrix -= matrix_mean
    matrix = matrix.multiply(matrix_stds)

    matrix_pad = (matrix !=0) * 3
    matrix += matrix_pad
    np.min(matrix)

    return matrix

'''

'''
def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])


save_sparse_csr('achats_households', X)

with open('households.csv','wb') as f:
    f.write(','.join(households))

with open('products.csv','wb') as f:
    f.write(','.join(products))
    

#    intruder_households = list(set(households) - set(list(households_original['household'])))
#    total_achats_households_new = total_achats_households.loc[intruder_households]
    
    total_achats_households = pd.merge(total_matrix, households_original, left_index=True, 
                                       right_on='household')
    total_achats_households['normalised_cons'] = total_achats_households[0] / total_achats_households['ucfo_val']
    
    total_achats_households['normalised_cons'] = np.round(total_achats_households['normalised_cons'] / 10,0) * 10
    total_achats_households['normalised_cons'] = total_achats_households['normalised_cons'].astype(int)
    #plot_histogram(total_achats_households, 'normalised_cons', 'Achats par foyers arrondis', reorder=True)
    
    
'''