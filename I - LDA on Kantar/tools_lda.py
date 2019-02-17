#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 10:34:33 2018

@author: andrei
"""


import pandas as pd
import numpy as np
import string
printable = set(string.printable)
import matplotlib.pyplot as  plt


#0 analyse de ka distribution des produits et sous produits


def clean_table(datatable):
    '''
    Cleans the datatable products
    drops the useless index and distinguishes btw 2 types of sousgroupes
    
    returns: cleaned datatable

    '''
    
    datatable = datatable.drop(['Unnamed: 0'], axis = 1)
    datatable = datatable.reset_index(drop=True)
    
    groupes = sorted(list(set(datatable['groupe'])))
    sousgroupes = sorted(list(set(datatable['sousgroupe'])))
    
    datatable['sousgroupe'].loc[(datatable['sousgroupe'] == sousgroupes[168]) & (datatable['groupe']== groupes[23])] = 'Pate_pate'
    datatable['sousgroupe'].loc[(datatable['sousgroupe'] == sousgroupes[168]) & (datatable['groupe']== groupes[8])] = 'Pate_charcuterie'
 
    return datatable


def transform_table_sousgroupe(subset, sousgroupe):
    '''
    transforms table
    the products are organized by sousgroupes
    this function creates subset tables that respect this organization
    '''
    
    headers = list(subset.columns.values)
    headers_question = [headers[i] for i in range(10,66,2)]
    headers_answer = [headers[i] for i in range(11,67,2)]
        

    headers_question_present = subset[headers_question].loc[0]

    list_indexes = list(headers_question_present.notnull())
    questions_present_index = [headers_question[i] for i in range(len(headers_question)) if list_indexes[i] == True]
    answers_present_index = [headers_answer[i] for i in range(len(headers_question)) if list_indexes[i] == True]
    
    subset_part1 = subset[headers[0:10]]
    subset_part2 = subset[answers_present_index]
    
    if len(questions_present_index) > 0:
        assert len(subset[questions_present_index].drop_duplicates()) == 1
    
    mapping = dict()
    for j in range(len(questions_present_index)):
        key = answers_present_index[j]
        value = headers_question_present[questions_present_index[j]]
        mapping[key] = filter(lambda x: x in printable, value)
    
    subset_part2 = subset_part2.rename(columns=mapping)
    subset_final = pd.merge(subset_part1, subset_part2, left_index=True, right_index=True)
    subset_final = subset_final.set_index('product')
    
    return subset_final




def clustering_algo_simple_sousgroupe(datatable, sousgroupe, cols_to_drop, full = True):
    '''
    L'algorithme de clusterisation
    Rassemble les produits qui ont les memes caracteristiques, apres avoir
        dropper certaines colonnes
    '''
    
    
    if full == True:
        subset = datatable.loc[datatable['sousgroupe'] == sousgroupe].reset_index(drop=True)
        subset = transform_table_sousgroupe(subset, sousgroupe)
        subset = subset.drop(cols_to_drop, axis=1)
    else:
        subset = datatable
        subset = subset.drop(cols_to_drop, axis=1)

    dict_products = dict()
    
    for index, row in subset.iterrows():
        descriptif_pre = [str(s) for s in list(row)]
        descriptif = ';'.join(descriptif_pre)
        
        try:
            dict_products[descriptif].append(index)
        except:
            dict_products[descriptif] = list()
            dict_products[descriptif].append(index)
    assert len(dict_products) == len(subset.drop_duplicates())
    
    list_final = list()
    for key in dict_products.keys():
        list_final.append(dict_products[key])
    
    return list_final

def clustering_algo_simple_all(datatable, sousgroupes, cols_to_drop, cluster_dict = None, advanced = False):
    '''
    fonction qui rassemble les differents clustering sousgroupes
        en un clustering du groupe total
    '''
    
    clustering_dict = dict()
    
    for sousgroupe in sousgroupes:
        clustering_list = list()
        if advanced == False:
            clustering_interm = clustering_algo_simple_sousgroupe(datatable, sousgroupe, cols_to_drop)
        else:
            clustering_interm = cluster_algo_drop_prot(datatable, cluster_dict, sousgroupe, cols_to_drop)
            
        for cluster in clustering_interm:
            clustering_list.append(cluster)
        clustering_dict[sousgroupe] = clustering_list
            
    return clustering_dict



def create_repesentative_table(products_table, cluster_dict, sousgroupe):
    '''
    fonction pour selectionner dans chaque cluster le produit representatif,
        i.e. le produit qui a été acheté le plus souvent
    '''
    
    list_repres = list()
    for cluster in cluster_dict[sousgroupe]:
        subset = list(products_table['count'].loc[products_table['product'].isin(cluster)])
        index_max = np.argmax(subset)
        list_repres.append(cluster[index_max])
    return list_repres

def count_clusters(clustering_map):
    counter = 0
    
    for key in clustering_map.keys():
        counter += len(clustering_map[key])
        
    return counter


def remove_headers_na(datatable, threshold):
    '''
    fonction pour enlever les variables non renseignees,
        tolerance de 50%
    '''
    subset_headers = list(datatable.columns.values)
    subset_headers = [hi for hi in subset_headers if hi not in cols_to_drop]
    subset_info_nan = datatable.isnull().sum().astype(float) / len(datatable)
    subset_info_nan_above50 = subset_info_nan.loc[subset_info_nan > threshold]
    headers_to_remove = list(subset_info_nan_above50.index.values)
    
    return headers_to_remove

def convert_float_to_quartiles(datatable):
    '''
    fonction pour convertir les colonnes numériques en quartiles, afin
        de faciliter le clustering
    '''
    types = datatable.dtypes
    float_types = types.loc[types=='float64']
    list_float_types = list(float_types.index.values)
    
    for float_header in list_float_types:
        datatable[float_header] = pd.qcut(datatable[float_header], 4, labels=False, duplicates = 'drop')
    
    return datatable

def select_column_drop(datatable):
    '''
    fonction pour enlever la colonne la plus différentiante,
        i.e. celle qui fait que si on l'enleve on peut regrouper le plus de produits
    '''
    list_values = list()
    headers = list(datatable.columns.values)
    
    for h in headers:
        list_values.append(len(datatable[[hi for hi in headers if hi != h]].drop_duplicates() ))
        
    minimum = np.argmin(list_values)
    return headers[minimum]
    

def cluster_algo_drop_prot(datatable, cluster_dict, sousgroupe, cols_to_drop):
    '''
    algorithme pour enlever les colonnes na
    '''
    
    subset = datatable.loc[datatable['product'].isin(create_repesentative_table(products_table, cluster_dict, sousgroupe))].reset_index(drop=True)
    subset = transform_table_sousgroupe(subset, sousgroupe)
    
    headers_to_remove = remove_headers_na(subset, 0.5)
    cols_to_drop_final = cols_to_drop[:]
    cols_to_drop_final.extend(headers_to_remove)
    
    subset = convert_float_to_quartiles(subset)
    
    #subset_temp = subset[[h for h in list(subset.columns.values) if h not in headers_to_remove]]
    #extra_col = select_column_drop(subset_temp) 
    #cols_to_drop_final.append(extra_col)
    new_clustering = clustering_algo_simple_sousgroupe(subset, sousgroupe, cols_to_drop_final, full = False)

    
    return new_clustering

 


def merge_clusters(clusters1, clusters2):
    
    '''
    algorithme pour melanger deux clustering différents
    '''
    
    dict_merge = dict()
    
    for key in clusters2.keys():

        dict_temp = dict()
        cluster_final = list()
        representative_list = create_repesentative_table(products_table, clusters1, key)
        
        for i in range(len(representative_list)):
            #assert representative_list[i] in clusters1[key][i]
            dict_temp[representative_list[i]] = clusters1[key][i]
        
        
        for cluster in clusters2[key]:
            cluster_temp = list()    
            
            for product in cluster:
                
                cluster_temp.extend(dict_temp[product])
            
            cluster_final.append(cluster_temp)
        
        dict_merge[key] = cluster_final
        
    return dict_merge
                
def compute_len_clusters(clusters_dict):
    '''
    algorithme pour mesurer la taille d'un clustering,
        i.e. combien de produits sont gardés
    '''
    len_counter = 0
    for key in clusters_dict:
        for cluster in clusters_dict[key]:
            len_counter += len(cluster)
    return len_counter


def create_pruned_datatable(datatable, cluster_dict):
    '''
    creer la table finale des cluster,
    un dictionnaire ou key est le produit representatif,
        value est un tuple [produits dans cluster, count total]
    '''
    dict_final = dict()
    for key in cluster_dict.keys():
        
        representative_list = create_repesentative_table(cluster_dict, key)
        dict_final[key] = dict()
        for i, cluster in enumerate(cluster_dict[key]):
            
            
            #assert representative_list[i] in cluster
            
            sum_counts = sum(datatable['count'].loc[datatable['product'].isin(cluster)])
            dict_final[key][representative_list[i]] = [cluster, sum_counts]
        
    return dict_final




def compute_total_count(dict_final):
    '''
    fonction pour calculer le count total d'un cluster,
        utile pour la fonction de creation de la table finale
    '''
    counter = 0
    for sousgroupe in dict_final.keys():
        for cluster_representative in dict_final[sousgroupe].keys():
            counter += dict_final[sousgroupe][cluster_representative][1]
    return counter

def pruned_clusters(cluster_dict_final, threshold, return_ratio =  False):
    '''
    selectionner seulement les clusters qui ont été achetés
        plus de fois qu'un certain nombre seuil
    '''
    cluster_final = dict()
    num = 0
    den = 0
    
    for sousgroupe in cluster_dict_final.keys():

        for cluster_representative in cluster_dict_final[sousgroupe].keys():
            den += cluster_dict_final[sousgroupe][cluster_representative][1]
            if(cluster_dict_final[sousgroupe][cluster_representative][1]) >= threshold:
                cluster_final[cluster_representative] = cluster_dict_final[sousgroupe][cluster_representative]
                num += cluster_dict_final[sousgroupe][cluster_representative][1]
    
    ratio = float(num) / den

    if return_ratio == False:
        return cluster_final
    else:
        return cluster_final, ratio
                



def create_dataframe(dict_final):
    '''
    transformer dictionnaire final en dataframe,
        pour que ce soit plus facile de l'enregistrer en csv
    '''
    list_to_return = list()
    
    for key in dict_final.keys():
        list_to_return.append([key, dict_final[key][0], dict_final[key][1]])
    
    dataframe = pd.DataFrame(list_to_return)
    return dataframe

def graph_len_products(x, y, title, name):
    plt.figure(figsize=(10,6), dpi=80)
    plt.plot(x, y, color="blue", linewidth=2.5, linestyle="-")
    plt.title(title)
    plt.savefig('lda_analyse/' + name,dpi=48)