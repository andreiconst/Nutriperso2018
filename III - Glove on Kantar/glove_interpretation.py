#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 15:16:03 2018

@author: andrei
"""

#interpretation of glove

import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

import sys
sys.path.insert(0, '/home/andrei/Desktop/kantar_2014/code_produits/')
from tools_preprocessing import *
from tools_clustering import *
import matplotlib.pyplot as plt

def create_simple_mapping(mapping_list):
    result = dict()
    for i, value in enumerate(mapping_list):
        result[value] = i
    return result

d = 300
embeddings = pd.read_csv('data_cleaned/embeddings_products_cp_mano.csv')
product_names = list(embeddings['Unnamed: 0'])
embeddings = np.asarray(embeddings[[str(i) for i in range(d)]])


products_table = pd.read_csv('data_cleaned/produits_achats.csv', encoding = 'latin1')
products_table = clean_table(products_table)
sousgroupes = sorted(list(products_table['sousgroupe'].drop_duplicates()))
dictionnaire_description = create_dict_to_keep(sousgroupes)
dictionnaire_index = create_simple_mapping(product_names)




def normalize_representation(representation):
    normalized_representation = np.zeros((representation.shape[0], representation.shape[1]))
    for i in range(representation.shape[0]):
        normalized_representation[i, :] = representation[i,:] / np.sqrt(np.dot(representation[i,:].T, representation[i,:]))
    return normalized_representation

def description_product(product_code, products_table, dictionnaire_description):
    row = products_table.loc[products_table['product'] == product_code].reset_index().loc[0]
    description = row['marque'] + '|' + row['sousgroupe'] + '|'
    description_header = dictionnaire_description[row['sousgroupe']][1:]
    
    for h in description_header:
        description += str(row[h]) +'|'
    return description

def find_k_closest(product_code, representation, product_names, products_table, dictionnaire_description, k = 10):
    
    center_product = description_product(product_code, products_table, dictionnaire_description)
    distances = np.ravel(np.dot(representation, representation[product_names.index(product_code),:]))
    
    indexes = np.argsort(-distances)
    result = list()
    for i in indexes[:k+1]:
        result.append(description_product(product_names[i], products_table, dictionnaire_description))
    
    return center_product, result

def analogy_game(product1, product2, product3, representation, product_names, products_table, dictionnaire_description, k = 10):
    descr_p1 = description_product(product1, products_table, dictionnaire_description)
    descr_p2 = description_product(product2, products_table, dictionnaire_description)
    descr_p3 = description_product(product3, products_table, dictionnaire_description)
    
    ersatz_product =  representation[product_names.index(product1),:] - representation[product_names.index(product2),:] + representation[product_names.index(product3),:]
    distances = np.ravel(np.dot(representation, ersatz_product))
    indexes = np.argsort(-distances)
    result = list()
    
    result.append(descr_p1 + ' - ' + descr_p2 + ' + ' + descr_p3 )
    for i in indexes[:k+1]:
        result.append(description_product(product_names[i], products_table, dictionnaire_description))
    
    return result


normalized_representation = normalize_representation(embeddings)


coef_products = pd.read_csv('data_cleaned/coef_regression.csv')
coef_products_nonzero = coef_products.loc[np.abs(coef_products['0']) > 0]
coef_products_zero = coef_products.loc[np.abs(coef_products['0']) == 0]


coef_products_nonzero['percentiles'] = pd.qcut(coef_products_nonzero['0'], 100, labels = False)
coef_products_zero['percentiles'] = -10
coef_products = pd.concat([coef_products_nonzero, coef_products_zero])

products_trimed = products_table.loc[products_table['product'].isin(product_names)]
products_trimed = pd.merge(products_trimed, coef_products, left_on = 'product', right_on = 'Unnamed: 0')
 


products_trimed_brsa = products_trimed.loc[products_trimed['sousgroupe'] == 'Brsa']
boissons = list(products_table['product'].loc[products_table['count'] > 100])
boissons_index = list()
for i in boissons:
    try:
        ind = product_names.index(i)
        boissons_index.append(i)
    except:
        pass
np.random.seed(1234)
_, test_neighbor = find_k_closest(29601, normalized_representation, product_names, products_table, 
                                            dictionnaire_description, k = 10)

indexes = np.random.randint(0, len(boissons_index), 20)
result = list()
for i in indexes:
    _, test_neighbor = find_k_closest(boissons_index[i], normalized_representation, product_names, products_table, 
                                            dictionnaire_description, k = 10)
    result.append(test_neighbor)
final_result = pd.DataFrame(result).T
#final_result.to_csv('results_glove/closest_neighbors_retrofitted.csv')

        
tsne = TSNE(n_components=2, random_state=123, perplexity = 30, metric = 'cosine')
tsne_repr = tsne.fit(normalized_representation)
representation = tsne_repr.embedding_
#tsne_representation = pd.DataFrame(representation, index = product_names)
#tsne_representation.to_csv('data_cleaned/tsne_representation_cluster50.csv')

table_representation = pd.DataFrame(representation, index = product_names )
products_trimed = pd.merge(products_trimed, table_representation, left_on = 'product', right_index = True)
products_trimed['decile'] = np.floor(products_trimed['percentiles']/10)


groupes = sorted(set(list(products_table['groupe'])))
groupes = [groupes[i] for i in [17,6,0,15,14,8,5,34,22,27,19,23,29,10,25,26,12,31,20,1]]



products_table_pruned = products_table.loc[products_table['mdd'] == 'Oui' ]
mdds = sorted(list(products_table_pruned['fabricant'].drop_duplicates()))
mdds = [mdds[i] for i in [0,8, 9,10,11,12,13,16,23]]
boissons = list(products_trimed['product'].loc[(products_trimed['count'] > 30) & (products_trimed['groupe'].isin(groupes)) & (products_trimed['percentiles'] != -10)])
boissons_index = list()
for i in boissons:
    try:
        boissons_index.append(product_names.index(i))
    except:
        pass

from tools_preprocessing import *




groupes = sorted(set(list(products_table['groupe'])))
groupes = [groupes[i] for i in [17,6,0,15,14,8,5,34,22,27,19,23,29,10,25,26,12,31,20,1]]



plot_by_group('100_size_TSNE_frequent_marques_noncoef_legend_mano.png', representation[boissons_index,:], 'groupe', products_trimed, [product_names[i] for i in boissons_index], with_legend = True)
plot_by_group('100_size_TSNE_frequent_marques_noncoef_nolegend_mano.png', representation[boissons_index,:], 'groupe', products_trimed, [product_names[i] for i in boissons_index], with_legend = False)

plot_by_group('00_size_TSNE_frequent_marques_coef_legend_mano.png', representation[boissons_index,:], 'decile', products_trimed, [product_names[i] for i in boissons_index], with_legend = True, hot_colors= True)
plot_by_group('00_size_TSNE_frequent_marques_coef_nolegend_mano.png', representation[boissons_index,:], 'decile', products_trimed, [product_names[i] for i in boissons_index], with_legend = False, hot_colors= True)


for groupe in groupes:
    boissons = list(products_table['product'].loc[(products_table['groupe'].isin([groupe])) &(products_table['count'] > 30)])
    boissons_index = list()
    for i in boissons:
        try:
            boissons_index.append(product_names.index(i))
        except:
            pass
    name = '22k_ssgrp_Retro_grp_Mano_TSNE_frequent_' + groupe + '_noncoef_nolegeng.png'
    plot_by_group(name, representation[boissons_index], 'sousgroupe', products_trimed, [product_names[i] for i in boissons_index], with_legend = False, hot_colors= False)
    name = '22k_ssgrp_Retro_grp_Mano_TSNE_frequent_' + groupe + '_noncoef_legeng.png'
    plot_by_group(name, representation[boissons_index], 'sousgroupe', products_trimed, [product_names[i] for i in boissons_index], with_legend = True, hot_colors= False)

for groupe in groupes:
    boissons = list(products_table['product'].loc[(products_table['groupe'].isin([groupe])) &(products_table['count'] > 30)])
    boissons_index = list()
    for i in boissons:
        try:
            boissons_index.append(product_names.index(i))
        except:
            pass
    name = '22k_ssgrp_Retro_grp_Mano_TSNE_frequent_' + groupe + '_coef_nolegeng.png'
    plot_by_group(name, representation[boissons_index], 'decile', products_trimed, [product_names[i] for i in boissons_index], with_legend = False, hot_colors= True)
    name = '22k_ssgrp_Retro_grp_Mano_TSNE_frequent_' + groupe + '_coef_legeng.png'
    plot_by_group(name, representation[boissons_index], 'decile', products_trimed, [product_names[i] for i in boissons_index], with_legend = True, hot_colors= True)




#
#test_representation = np.dot(embeddings, embeddings.T)
#test_index = np.argsort(-test_representation[123,:])[:10]
#closest_product_test = [product_names[ind] for ind in test_index]
#table_test = products_trimed.loc[products_trimed['product'].isin(closest_product_test)]
#
#
#u = embeddings[product_names.index(73290), :]
#v = embeddings[product_names.index(223128), :]
#np.dot(u - v*0.1, v.T) 
#loss = 0.6 - 1 + 0.8694948480421248





def extract_products_cluster_TSNE(representation, col_names, products_trimed, x_inf, x_sup, y_inf, y_sup, groupe = ''):
    if groupe == '':
        representation_final = representation
        col_final = col_names
    else:
        col_products = products_trimed['product'].loc[products_trimed['groupe'] == groupe]
        index = list()
        for i in col_products:
            index.append(col_names.index(i))
        index = sorted(index)
        representation_final = representation[index,:]
        col_final = [col_names[i] for i in index]
    
    indexes = np.where((representation_final[:,0] > x_inf) & (representation_final[:,0] < x_sup) & (representation_final[:,1] > y_inf) & (representation_final[:,1] < y_sup))[0]
    
    final_prods = [col_final[i] for i in indexes]
    result = products_trimed.loc[products_trimed['product'].isin(final_prods)]
    return result

#Analyse pain et viennoiserie
boisson_1 = extract_products_cluster_TSNE(representation, product_names, products_trimed, -100, 100, -100, 0, groupe = 'Boisson')
boisson_2 = extract_products_cluster_TSNE(representation, product_names, products_trimed, -100, 100, 0, 100, groupe = 'Boisson')



#Analyse aide culinaire
aide_1 = extract_products_cluster_TSNE(representation, product_names, products_trimed, 60, 100, 0, 100, groupe = 'Aide Culinaire')
aide_2 = extract_products_cluster_TSNE(representation, product_names, products_trimed, 60, 100, -90, 0, groupe = 'Aide Culinaire')


#Analyse des boissons

boissons_1 = extract_products_cluster_TSNE(representation, product_names, products_trimed, -60, -30, -20, 10, groupe = 'Boisson')
boissons_2 = extract_products_cluster_TSNE(representation, product_names, products_trimed, -60, 100, 40, 60, groupe = 'Boisson')
boissons_3 = extract_products_cluster_TSNE(representation, product_names, products_trimed, -20, 0, 0, 20, groupe = 'Boisson')
boissons_4 = extract_products_cluster_TSNE(representation, product_names, products_trimed, 10, 30, 0, 20, groupe = 'Boisson')

#Analyse des fromages

fromage1 = extract_products_cluster_TSNE(representation, product_names, products_trimed, -30, -15, -20, 0, groupe = 'Fromage')
fromage2 = extract_products_cluster_TSNE(representation, product_names, products_trimed, -30, -20, 5, 40, groupe = 'Fromage')

#Plat frais

plats1 = extract_products_cluster_TSNE(representation, product_names, products_trimed, 10, 25, 20, 100, groupe = 'Plat')
plats2 = extract_products_cluster_TSNE(representation, product_names, products_trimed, 30, 100, 20, 100, groupe = 'Plat')
plats3 = extract_products_cluster_TSNE(representation, product_names, products_trimed, -20, 0, 40, 100, groupe = 'Plat')
plats4 = extract_products_cluster_TSNE(representation, product_names, products_trimed, -20, 0, -10, 10, groupe = 'Plat')

#Fruits et legumes

fruits_leg1 = extract_products_cluster_TSNE(representation, product_names, products_trimed, -100, 100, 35, 100, groupe = 'Fruits et Legumes')
fruits_leg2 = extract_products_cluster_TSNE(representation, product_names, products_trimed, -20, 0, -20, 0, groupe = 'Fruits et Legumes')
fruits_leg3 = extract_products_cluster_TSNE(representation, product_names, products_trimed, 15, 25, -100, -30, groupe = 'Fruits et Legumes')

#general

general1 = extract_products_cluster_TSNE(representation, product_names, products_trimed, 30, 50, 25, 50, groupe = '')
general2 = extract_products_cluster_TSNE(representation, product_names, products_trimed, 10, 20, -18, 6, groupe = '')
general3 = extract_products_cluster_TSNE(representation, product_names, products_trimed, -30, -20, 6, 20, groupe = '')



####
# Testing the clusters

def delta_integrales_discrete(array1, array2):
    
    result = 0
    
    min_value = np.min([np.min(array1), np.min(array2)])
    max_value = np.max([np.max(array1), np.max(array2)])
    
    result = 0
    for i in range(int(min_value)-1, int(max_value) +1):
        result += (np.sum(array1 < i)/len(array1)) - (np.sum(array2 < i) /len(array2))
    
    return result


kmeans = KMeans(n_clusters = 50, random_state = 123)
clusters = kmeans.fit(normalized_representation)
clusters_appartenance = clusters.labels_

cluster_dict_index = dict()
cluster_dict = dict()

for i, cl in enumerate(clusters_appartenance):
    try:
        cluster_dict[cl].append(product_names[i])
        cluster_dict_index[cl].append(i)
    except:
        cluster_dict[cl] = list()
        cluster_dict[cl].append(product_names[i])
        cluster_dict_index[cl] = list()
        cluster_dict_index[cl].append(i)
    
result = list()
np.random.seed(123)
random_subset = np.random.randint(0, purchase_matrix_clustered.shape[0], 1000)
for cl in cluster_dict.keys():
    ratio = np.sum(purchase_matrix_clustered[:,cluster_dict_index[cl]], axis= 1) / np.sum(purchase_matrix_clustered, axis= 1)
    most_index = np.argsort(-np.ravel(ratio))[:1000]
#    least_index = np.argsort(np.ravel(ratio))[:1000]
    families_readable = [int(row_names_original[i]) for i in most_index]
#    families_unreadable = [int(row_names_original[i]) for i in least_index]
    families_random = [int(row_names_original[i]) for i in list(set(random_subset) - set(most_index))]
    
    households_concerned = households.loc[families_readable].dropna(subset=['bmi'])
    households_random = households.loc[families_random].dropna(subset=['bmi'])
    r = delta_integrales_discrete(np.asarray(households_concerned['bmi']), np.asarray(households_random['bmi']))
#    ratio_fat_1 = len(households_concerned.loc[households_concerned['bmi'] > 25]) / len(households_concerned)
#    ratio_fat_2 = len(households_concerned.loc[households_concerned['bmi'] > 30]) / len(households_concerned)

    result.append([cl, r])

prod_ben1 = products_trimed.loc[products_trimed['product'].isin(cluster_dict[45])]
prod_ben2 = products_trimed.loc[products_trimed['product'].isin(cluster_dict[16])]
prod_ben3 = products_trimed.loc[products_trimed['product'].isin(cluster_dict[17])]
prod_ben4 = products_trimed.loc[products_trimed['product'].isin(cluster_dict[14])]



prod_mal1 = products_trimed.loc[products_trimed['product'].isin(cluster_dict[46])]
prod_mal2 = products_trimed.loc[products_trimed['product'].isin(cluster_dict[30])]
prod_mal3 = products_trimed.loc[products_trimed['product'].isin(cluster_dict[91])]

prod_mal2.to_csv('meat_prodcuts.csv')
prod_ben1.to_csv('vegetable_fruits_prodcuts.csv')

#Are bio people healthier?
## Answer: YES !! YAYYYYYY





bio_products = list(general3['product'])
col_bio = list()
for i in bio_products:
    col_bio.append(col_names_clustered.index(i))
bio_ratio = np.sum(purchase_matrix_clustered[:,col_bio], axis= 1) / np.sum(purchase_matrix_clustered, axis= 1)
bio_ratio[np.isnan(bio_ratio)] = 0
plt.hist(bio_ratio[:,0])

bio_families = np.where(bio_ratio > .15)[0]
bio_families_readable = [int(row_names_original[i]) for i in bio_families]

households_bio = households.loc[bio_families_readable]
ratio_fat_bio1 = len(households_bio.loc[households_bio['bmi'] > 25]) / len(households_bio)
ratio_fat_bio2 = len(households_bio.loc[households_bio['bmi'] > 30]) / len(households_bio)


ratio_fat_pop1 = len(households.loc[households['bmi'] > 25]) / len(households)
ratio_fat_pop2 = len(households.loc[households['bmi'] > 30]) / len(households)


households_bio_sum = households_bio.describe()


#Are sodas bad ?
##
sodas = extract_products_cluster_TSNE(representation, product_names, products_trimed, -35, -25, 0, 10, groupe = '')

soda_products = list(sodas['product'])
col_soda = list()
for i in soda_products:
    col_soda.append(col_names_clustered.index(i))
soda_ratio = np.sum(purchase_matrix_clustered[:,col_soda], axis= 1) / np.sum(purchase_matrix_clustered, axis= 1)
soda_ratio[np.isnan(soda_ratio)] = 0
plt.hist(soda_ratio[:,0])

soda_families = np.where(soda_ratio > .15)[0]
soda_families_readable = [int(row_names_original[i]) for i in soda_families]

households_soda = households.loc[soda_families_readable]
ratio_fat_soda1 = len(households_soda.loc[households_soda['bmi'] > 25]) / len(households_soda)
ratio_fat_soda2 = len(households_soda.loc[households_soda['bmi'] > 30]) / len(households_soda)



households_bio_sum = households_bio.describe()





products_table_mdd = products_table.loc[products_table['mdd'] == 'Oui']
products_table_mdd_trimmed = products_table.drop(columns=['product']).drop_duplicates()



test_analogy = analogy_game(normalized_representation, product_names, products_table, 
                                            dictionnaire_description, k = 10)