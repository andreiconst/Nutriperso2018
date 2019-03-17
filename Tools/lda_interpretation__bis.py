#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 14:46:50 2018

@author: andrei
"""


import pandas as pd
import numpy as np
import csv


import sys
sys.path.insert(0, '/home/andrei/Desktop/kantar_2014/code_produits/')

from tools_lda import *
from tools_clustering import *

'''
Input:
    1. name products
    2. name households
    3. topic table (products x topics)
    4. topic distribution table (households x topics)

'''


# =============================================================================
# link topics with demographic variables (heatmap)
# =============================================================================

def create_household_index_dictionary(list_households):
    
    dict_household_index = dict()
    
    for i in range(len(list_households)):
        dict_household_index[list_households[i]] = i
        
    return dict_household_index

def compute_household_variable_mean_lda(topic_index, variable, datatable_lda, datatable_households, index_households):
    
    dict_household_index = create_household_index_dictionary(index_households)
    
    variable_values = list(datatable_households[variable].drop_duplicates())
    mean_list = list()
    
    mean = np.mean(np.asarray(datatable_lda[:, topic_index]))
    std = np.std(np.asarray(datatable_lda[:,topic_index]))
    
    for var in variable_values:
        households_of_interest = datatable_households.loc[datatable_households[variable] == var]
        households_of_interest = list(households_of_interest['household'])
        
        indexes_of_interest = [dict_household_index[h] for h in households_of_interest]
        
        mean_value = np.mean(np.asarray(datatable_lda[indexes_of_interest,topic_index]))
        
        mean_list.append([var, (mean_value - mean) / std])
    
    result = pd.DataFrame(mean_list)
    return result
        

def create_dict_column(column):
    liste_produits = list(lda_topics.index.values)
    dict_sousgroupe = dict()
    column_content = list(products_table[column].drop_duplicates())
    for cont in column_content:
        
        liste_sousgroupe = list(products_table['product'].loc[products_table[column]==cont])
        index_sousgroupe = [liste_produits.index(liste_sousgroupe[i]) for i in range(len(liste_sousgroupe))]
        dict_sousgroupe[cont] = index_sousgroupe
    
    return dict_sousgroupe


def compute_value_matrix(lda_table, dict_col,total_topics,total_products):
    
    sousgroupe_in_cluster = np.zeros((len(dict_col), len(total_topics)))
    cluster_in_sougroupe = np.zeros((len(dict_col), len(total_topics)))
    q_prop = np.zeros((len(dict_col), len(total_topics)))

    keys = list(dict_col.keys())
    for i, sousgroupe in enumerate(keys):
        
        sousgroupe_in_cluster[i,:] = np.sum(lda_table[:, dict_col[sousgroupe]], axis=1) / total_topics
        cluster_in_sougroupe[i,:] = np.sum(lda_table[:, dict_col[sousgroupe]], axis=1) / np.sum(total_products[dict_col[sousgroupe]])
        q_prop[i,:] = (np.sum(lda_table[:, dict_col[sousgroupe]], axis=1) / total_topics) / (np.sum(total_products[dict_col[sousgroupe]]) / np.sum(total_topics))
    
    return keys, sousgroupe_in_cluster, cluster_in_sougroupe,q_prop
    

def represent_sousgroupe(lda_table, max_dim=10):
    liste_produits = list(lda_topics.index.values)

    dict_sousgroupe = create_dict_column('sousgroupe')
    dict_mdd = create_dict_column('mdd')
    dict_bio = create_dict_column('bio')
    
    lda_table = np.asarray(lda_table).T
    total_topics = np.sum(lda_table, axis = 1)
    total_products = np.sum(lda_table, axis = 0)
    sousgroupes, s_in_cl, cl_in_s, q_table = compute_value_matrix(lda_table,dict_sousgroupe,total_topics, total_products)
    bio, s_in_clbio, cl_in_sbio, q_tablebio =  compute_value_matrix(lda_table,dict_bio,total_topics, total_products)
    mdd, s_in_clmdd, cl_in_smdd, q_tablemdd = compute_value_matrix(lda_table,dict_mdd,total_topics, total_products)
 
    list_final = list()
    for i in range(nb_topics):
        list_interm = list()
        value = 0
        value_total = np.sum(lda_table[i,:])

        indexes_level = np.argsort(-lda_table[i,:])[:max_dim]
        for index in indexes_level:
            
            v = lda_table[i, index] / value_total
            row = products_table.loc[products_table['product'] == liste_produits[index]]
            description = list(row[dictionary_description[row['sousgroupe'].values[0]]].values[0][1:])
            description.insert(0, row['sousgroupe'].values[0])
            description = [str(d) for d in description]
            list_interm.append(' '.join(description) + ' : %.4f' % v) 
            value+=v

        list_interm.append('')
        
        indexes_level = np.argsort(-q_table[:,i])[:max_dim]
        for index in indexes_level:
            list_interm.append(sousgroupes[index] + ' : %.2f | %.2f | %.2f' % (s_in_cl[index,i], q_table[index,i], cl_in_s[index,i]) )
        list_interm.append('')
        
        indexes_level = np.argsort(-s_in_cl[:,i])[:max_dim]   
        for index in indexes_level:
            list_interm.append(sousgroupes[index] + ' : %.2f | %.2f | %.2f' % (s_in_cl[index,i], q_table[index,i], cl_in_s[index,i])) 
        list_interm.append('')

        
        list_interm.append('total n :' + str(np.round(value,2)))
        list_interm.append('bio : ' +  str(np.round(s_in_clbio[bio.index('Oui'),i],2)) + ' | ' +  str(np.round(cl_in_sbio[bio.index('Oui'),i],2)) + ' | ' +  str(np.round(q_tablebio[bio.index('Oui'),i],2)))
        list_interm.append('mdd : ' +  str(np.round(s_in_clmdd[mdd.index('Oui'),i],2)) + ' | ' +  str(np.round(cl_in_smdd[mdd.index('Oui'),i],2)) + ' | ' +  str(np.round(q_tablemdd[mdd.index('Oui'),i],2)))
            
        list_final.append(list_interm)
        
    return pd.DataFrame(list_final).transpose()

#table_sousgroupes = represent_sousgroupe(lda_topics, max_dim=20)
#table_sousgroupes.to_csv('Reunions/sousgroupes_lda_topics.csv', index=False)
