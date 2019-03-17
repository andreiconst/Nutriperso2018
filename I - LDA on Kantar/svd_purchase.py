#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 17:42:56 2018

@author: andrei
"""

#LSA on purchase data
import pandas as pd
import numpy as np
import csv
import math
from scipy.sparse.linalg import svds, norm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


import sys
sys.path.insert(0, '/home/andrei/Desktop/kantar_2014/code_produits/')
sys.path.insert(0, '/home/andrei/Desktop/kantar_2014/code_foyers/')
from tools_preprocessing import *
from foyers_processing import *
from tools_clustering import *


purchase_matrix, row_names_original, col_names_original = load_purchase_matrix('data/purchase_table_full.npz', 
                                                                 'data/households_matrix_full.txt', 
                                                                 'data/products_matrix_full.txt')


products_table = pd.read_csv('data/produits_achats.csv', encoding='latin1')
#cluster_table = pd.read_csv('data_cleaned/cluster_products_mano.csv', encoding='latin1')
cluster_table = pd.read_csv('data/cluster_products_mano.csv', encoding='latin1') #the other clustering proposed


purchase_matrix_clustered, col_names_clustered = cluster_products(cluster_table, products_table, purchase_matrix, col_names_original)


households = pd.read_csv('data_cleaned/households_trimed.csv')
households = insert_regions(households)
household_names = list(households['household'])
household_names = [int(h) for h in household_names]
households = households.set_index('household')
households['n_enf'] = households['en3'] + households['en6'] + households['en15'] + households['en25']

households['situation'] = 'couple'
households['situation'].loc[(households['is_pere'] == True) & (households['is_mere'] == False) ] = 'single man'
households['situation'].loc[(households['is_pere'] == False) & (households['is_mere'] == True) ] = 'single woman'


households['age_pere'].loc[households['age_pere'].isnull()] = 0
households['age_mere'].loc[households['age_mere'].isnull()] = 0
households['age_total'] = households['age_pere'] + households['age_mere']
households['taille_foyer'] = households['is_mere'].astype(int) + households['is_pere'].astype(int)
households['age_moyen']= households['age_total'] / households['taille_foyer']
households['age_arrondi'] = np.round(households['age_moyen'] / 5,0) * 5

households['cha'].loc[households['cha'].isnull()] = 0
households['cha'].loc[households['cha'] >= 3] = 3

households['chie'].loc[households['chie'].isnull()] = 0
households['chie'].loc[households['chie'] >= 3] = 3

households['etud_pere'].loc[households['etud_pere'].isnull()] = 0
households['etud_mere'].loc[households['etud_mere'].isnull()] = 0

households['etud_max'] = households[["etud_mere", "etud_pere"]].max(axis=1)



d = 300
embeddings = pd.read_csv('data_cleaned/embeddings_products_cp_mano.csv')
product_names = list(embeddings['Unnamed: 0'])

index_col = list()
for i, prod in enumerate(product_names):
    index_col.append(col_names_clustered.index(prod))




purchase_matrix_clustered = purchase_matrix_clustered[:,index_col]
col_names_clustered = [col_names_clustered[i] for i in index_col]
purchase_matrix_clustered.data = np.log(purchase_matrix_clustered.data + 1)

purchase_matrix_final, row_names_final= delete_rows_entropy(purchase_matrix_clustered, row_names_original, 4)
#purchase_matrix_final, row_names_final= purchase_matrix_normalized, row_names_normalized
purchase_matrix_total = np.sum(purchase_matrix_final, axis= 1)
purchase_matrix_final = purchase_matrix_final.multiply(1/purchase_matrix_total)
purchase_matrix_final = purchase_matrix_final.tocsr()

mask_hh = list()
for i,h in enumerate(row_names_final):
    if int(h) in household_names:
        mask_hh.append(i)
        
        
purchase_matrix_final = purchase_matrix_final[mask_hh,:]
row_names_touse = [int(row_names_final[i]) for i in mask_hh]



sum(households['regionName'].isnull().values)

u, s, vt = svds(purchase_matrix_final, 300)
svd_households = u * s

regions = list(set(households['regionName']))
regions = sorted(regions)

tsne = TSNE(n_components=2, random_state=123, perplexity = 30)
tsne_repr = tsne.fit(svd_households)
representation = tsne_repr.embedding_

plot_by_group_simple('households_lsa_scla.png', representation, 'scla', households, row_names_touse, with_legend = True, hot_colors=True)
plot_by_group_simple('households_lsa_nenf.png', representation, 'n_enf', households, row_names_touse, with_legend = True, hot_colors=False)
plot_by_group_simple('households_lsa_bb.png', representation, 'en3', households, row_names_touse, with_legend = True, hot_colors=True)
plot_by_group_simple('households_lsa_region.png', representation, 'regionName', households, row_names_touse, with_legend = True, hot_colors=False)
plot_by_group_simple('households_lsa_situation.png', representation, 'situation', households, row_names_touse, with_legend = True, hot_colors=False)
plot_by_group_simple('households_lsa_age.png', representation, 'age_arrondi', households, row_names_touse, with_legend = True, hot_colors=True)
plot_by_group_simple('households_lsa_cha.png', representation, 'cha', households, row_names_touse, with_legend = True, hot_colors=True)
plot_by_group_simple('households_lsa_chie.png', representation, 'chie', households, row_names_touse, with_legend = True, hot_colors=True)
plot_by_group_simple('households_lsa_etud.png', representation, 'etud_max', households, row_names_touse, with_legend = True, hot_colors=True)

###############################################################################

#Analysis on products

svd_prods = vt.T * s

tsne = TSNE(n_components=2, random_state=123, perplexity = 30)
tsne_repr = tsne.fit(svd_prods)
representation = tsne_repr.embedding_


products_table = pd.read_csv('data_cleaned/produits_achats.csv', encoding = 'latin1')
products_table = clean_table(products_table)

products_trimed = products_table.loc[products_table['product'].isin(product_names)]
products_trimed = products_trimed.set_index('product')
plot_by_group_simple('products_lsa_legend.png', representation, 'groupe', products_trimed, product_names, with_legend = True)
plot_by_group_simple('products_lsa_nolegend.png', representation, 'groupe', products_trimed, product_names, with_legend = False)

