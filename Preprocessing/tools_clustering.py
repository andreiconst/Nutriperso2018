#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 11:51:14 2018

@author: andrei
"""

import pandas as pd
import numpy as np
import string
printable = set(string.printable)


cols_to_drop = ['fabricant', 'marque']

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


def transform_table_sousgroupe(subset, sousgroupe, show_title = False):
    '''
    transforms table
    the products are organized by sousgroupes
    this function creates subset tables that respect this organization
    '''
    
    headers = list(subset.columns.values)
    headers_question = [headers[i] for i in range(10,66,2)]
    headers_answer = [headers[i] for i in range(11,67,2)]
        

    headers_question_present = subset[headers_question].reset_index().loc[0]

    list_indexes = list(headers_question_present.notnull())[1:]
    questions_present_index = [headers_question[i] for i in range(len(headers_question)) if list_indexes[i]]
    answers_present_index = [headers_answer[i] for i in range(len(headers_question)) if list_indexes[i]]
    
    hearders1 = headers[0:10]
    hearders1.append(headers[len(headers) - 1])
    subset_part1 = subset[hearders1]
    subset_part2 = subset[answers_present_index]
    
    if len(questions_present_index) > 0:
        assert len(subset[questions_present_index].drop_duplicates()) == 1
    
    mapping = dict()
    for j in range(len(questions_present_index)):
        
        key = answers_present_index[j]
        value = headers_question_present[questions_present_index[j]]
#        mapping[key] = filter(lambda x: x in printable, value)
        mapping[key] = value

    if show_title == True:
        subset_part2 = subset_part2.rename(columns=mapping)
    subset_final = pd.merge(subset_part1, subset_part2, left_index=True, right_index=True)
    subset_final = subset_final.set_index('product')
    subset_final = convert_float_to_quartiles(subset_final)
    
    return subset_final


def clustering_algo_simple_sousgroupe(datatable, sousgroupe, cols_to_drop, dict_qval, full = True):
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

    dict_clusters = dict()
    dict_conditions = dict()
    counter = 0
    
    
    for product, row in subset.iterrows():
        
        val_q = dict_qval[product]
        descriptif_pre = [str(s) for s in list(row)]
        descriptif = ';'.join(descriptif_pre)
        
        try:
            to_continue = True
            for value in dict_conditions[descriptif]:
                val_ref = value[0]
                index_cluster = value[1]
                if (val_q >= val_ref * 0.2) and (val_q <= val_ref * 2) :
                    dict_clusters[index_cluster].append(product)
                    to_continue = False
                    break
            
            if to_continue == True:
                counter +=1
                dict_conditions.append([val_q, counter])
                dict_clusters[counter] = list()
                dict_clusters[counter].append(product)

        except:
            counter += 1
            dict_conditions[descriptif] = list()
            dict_conditions[descriptif].append([val_q, counter])
            dict_clusters[counter] = list()
            dict_clusters[counter].append(product)
#    assert len(dict_products) == len(subset.drop_duplicates())
    
    list_final = list()
    for key in dict_clusters.keys():
        list_final.append(dict_clusters[key])
    
    return list_final


def clustering_algo_mano(datatable, sousgroupe, dict_cols_keep, dict_qval, full = True):
    '''
    L'algorithme de clusterisation
    Rassemble les produits qui ont les memes caracteristiques, apres avoir
        dropper certaines colonnes
    '''
    
    if full == True:
        subset = datatable.loc[datatable['sousgroupe'] == sousgroupe].reset_index(drop=True)
        subset = transform_table_sousgroupe(subset, sousgroupe)
        subset = subset[dict_cols_keep[sousgroupe]]
    else:
        subset = datatable
        subset = subset[dict_cols_keep[sousgroupe]]

    dict_clusters = dict()
    dict_conditions = dict()
    counter = 0
    
    for product, row in subset.iterrows():
        
        val_q = dict_qval[product]
        val_q = 0
        descriptif_pre = [str(s) for s in list(row)]
        descriptif = ';'.join(descriptif_pre)
        
        try:
            to_continue = True
            for value in dict_conditions[descriptif]:
                val_ref = value[0]
                index_cluster = value[1]
                if (val_q >= val_ref * 0.1) and (val_q <= val_ref * 3) :
                    dict_clusters[index_cluster].append(product)
                    to_continue = False
                    break
            
            if to_continue == True:
                counter +=1
                dict_conditions.append([val_q, counter])
                dict_clusters[counter] = list()
                dict_clusters[counter].append(product)

        except:
            counter += 1
            dict_conditions[descriptif] = list()
            dict_conditions[descriptif].append([val_q, counter])
            dict_clusters[counter] = list()
            dict_clusters[counter].append(product)
#    assert len(dict_products) == len(subset.drop_duplicates())
    
    list_final = list()
    for key in dict_clusters.keys():
        list_final.append(dict_clusters[key])
    
    return list_final



def clustering_algo_full_mano(datatable, sousgroupes, dict_cols_keep, dict_qval):
    '''
    fonction qui rassemble les differents clustering sousgroupes
        en un clustering du groupe total
    '''
    clustering_dict = dict()
    
    for sousgroupe in sousgroupes:
        clustering_list = list()
        clustering_interm = clustering_algo_mano(datatable, sousgroupe, dict_cols_keep,dict_qval)

            
        for cluster in clustering_interm:
            clustering_list.append(cluster)
        clustering_dict[sousgroupe] = clustering_list
            
    return clustering_dict



def clustering_algo_simple_all(datatable, sousgroupes, cols_to_drop, dict_qval, cluster_dict = None, advanced = False):
    '''
    fonction qui rassemble les differents clustering sousgroupes
        en un clustering du groupe total
    '''
    
    clustering_dict = dict()
    
    for sousgroupe in sousgroupes:
        
        clustering_list = list()
        if advanced == False:
            clustering_interm = clustering_algo_simple_sousgroupe(datatable, sousgroupe, cols_to_drop, dict_qval)
        else:
            clustering_interm = cluster_algo_drop_prot(datatable, cluster_dict, sousgroupe, cols_to_drop, dict_qval)
            
        for cluster in clustering_interm:
            clustering_list.append(cluster)
        clustering_dict[sousgroupe] = clustering_list
            
    return clustering_dict


def count_clustering_numbers(clustering, datatable, table_centrale):
    result = list()
    index_list = list()
    for cluster in clustering:
        temp = np.zeros(12)
        for prod in cluster:
            temp += table_centrale[prod]

        count = sum(datatable['count'].loc[datatable['product'].isin(cluster)])
        temp /= np.sum(temp)
        result.append([count, cluster, entropy(temp), temp.tolist()])
        index_list.append(count)
    index = np.argsort(-np.asarray(index_list))
    return [result[i] for i in index]



def create_repesentative_table(products_table, cluster_dict, sousgroupe):
    '''
    fonction pour selectionner dans chaque cluster le produit representatif,
        i.e. le produit qui a été acheté le plus souvent
    '''
    
    list_repres = list()
    for cluster in cluster_dict[sousgroupe]:
        subset = products_table['count'].loc[products_table['product'].isin(cluster)]
        index_max = subset.argmax()
        list_repres.append(products_table['product'].loc[index_max])
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
    

def cluster_algo_drop_prot(datatable, cluster_dict, sousgroupe, cols_to_drop, dict_qval):
    '''
    algorithme pour enlever les colonnes na
    '''
    
    subset = datatable.loc[datatable['product'].isin(create_repesentative_table(datatable, cluster_dict, sousgroupe))].reset_index(drop=True)
    subset = transform_table_sousgroupe(subset, sousgroupe)
    
    headers_to_remove = remove_headers_na(subset, 0.5)
    cols_to_drop_final = cols_to_drop[:]
    cols_to_drop_final.extend(headers_to_remove)
    
    subset = convert_float_to_quartiles(subset)
    
    #subset_temp = subset[[h for h in list(subset.columns.values) if h not in headers_to_remove]]
    #extra_col = select_column_drop(subset_temp) 
    #cols_to_drop_final.append(extra_col)
    new_clustering = clustering_algo_simple_sousgroupe(subset, sousgroupe, cols_to_drop_final, dict_qval, full = False)

    
    return new_clustering

 
def merge_clusters(datatable, clusters1, clusters2):
    
    '''
    algorithme pour melanger deux clustering différents
    '''
    
    dict_merge = dict()
    
    for key in clusters2.keys():

        dict_temp = dict()
        cluster_final = list()
        representative_list = create_repesentative_table(datatable, clusters1, key)
        
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
        
        representative_list = create_repesentative_table(datatable, cluster_dict, key)
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


def create_dict_val_qvol(products_table):
    result = dict()
    for i, row in products_table.iterrows():
        result[row['product']] = row['valqvol']
    return result

def create_table_centrale(full_table, products_table, centrales):
    
    centrale_aggr = full_table[['product', 'centraleachat', 'codepanier']].groupby(['product', 'centraleachat']).count().reset_index()
    result = dict()
    
    counter = 0
    for index, row in centrale_aggr.iterrows():
        try:
            result[row['product']][centrales.index(row['centraleachat'])] = row['codepanier']
        except:
            result[row['product']] = np.zeros(len(centrales))
            
            try:
                result[row['product']][centrales.index('centraleachat')] = row['codepanier']
            except:
                pass
        counter +=1
        
        if counter % 10000 == 0:
            print(counter)
    return result



def entropy(data):
#    assert np.sum(data) == 1
    assert data.all() >= 0
    data = data[data > 0]
    entropy = -np.dot(data, np.log(data))
    return entropy

def create_dict_to_keep(sousgroupes):
    
    dict_to_keep = dict({sousgroupes[0] : ['codeqvol', 'mdd', 'bio','v_6', 'v_7'],
                         sousgroupes[1] : ['codeqvol', 'mdd', 'bio','v_7',],
                         sousgroupes[2] : ['codeqvol', 'mdd', 'bio','v_6', 'v_9', 'v_10'],
                         sousgroupes[3] : ['codeqvol', 'mdd', 'bio','v_7', 'v_8',],
                         sousgroupes[4] : ['codeqvol', 'mdd', 'bio','v_8',],
                         sousgroupes[5] : ['codeqvol', 'mdd', 'bio','v_5',],
                         sousgroupes[6] : ['codeqvol', 'mdd', 'bio','v_7', 'v_10',],
                         sousgroupes[7] : ['codeqvol','mdd', 'bio',],
                         sousgroupes[8] : ['codeqvol', 'mdd', 'bio','v_4',],
                         sousgroupes[9] : ['codeqvol','mdd', 'bio',],
                         sousgroupes[10] : ['codeqvol', 'mdd', 'bio','vn_1'],
                         sousgroupes[11] : ['codeqvol', 'mdd', 'bio','v_7', 'v_8',],
                         sousgroupes[12] : ['codeqvol', 'mdd', 'bio',],
                         sousgroupes[13] : ['codeqvol', 'mdd', 'bio','vn_3', 'v_5', 'v_6',],
                         sousgroupes[14] : ['codeqvol', 'mdd', 'bio', 'v_4',],
                         sousgroupes[15] : ['codeqvol', 'mdd', 'bio','v_5',],
                         sousgroupes[16] : ['codeqvol', 'mdd', 'bio','v_4',],
                         sousgroupes[17] : ['codeqvol', 'bio'],
                         sousgroupes[18] : ['codeqvol', 'mdd', 'bio','v_4',],
                         sousgroupes[19] : ['codeqvol', 'mdd', 'bio',],
                         sousgroupes[20] : ['codeqvol', 'mdd', 'v_4','v_7',],
                         sousgroupes[21] : ['codeqvol', 'mdd', 'bio','v_4',],
                         sousgroupes[22] : ['codeqvol', 'mdd', 'bio','v_4',],
                         sousgroupes[23] : ['codeqvol', 'mdd', 'bio','v_5',],
                         sousgroupes[24] : ['codeqvol', 'mdd', 'bio','v_4',],
                         sousgroupes[25] : ['codeqvol',],
                         })
    
        
        
    
    dict_to_keep[sousgroupes[26]] = ['codeqvol', 'mdd', 'bio',]   
    dict_to_keep[sousgroupes[27]] = ['codeqvol', 'mdd', 'bio','v_7']   
    dict_to_keep[sousgroupes[28]] = ['codeqvol', 'mdd', 'bio','v_5']   
    dict_to_keep[sousgroupes[29]] = ['codeqvol', 'mdd', 'bio','v_4']   
    dict_to_keep[sousgroupes[30]] = ['codeqvol', 'mdd', 'bio']
       
    dict_to_keep[sousgroupes[31]] = ['codeqvol', 'mdd', 'bio', 'v_4', 'v_11', 'v_14']   
    dict_to_keep[sousgroupes[32]] = ['codeqvol', 'mdd', 'bio', 'v_5']   
    dict_to_keep[sousgroupes[33]] = ['codeqvol', 'mdd', 'bio']   
    dict_to_keep[sousgroupes[34]] = ['codeqvol', 'mdd', 'v_7', 'v_8']   
    dict_to_keep[sousgroupes[35]] = ['codeqvol', 'mdd', 'bio']   
    dict_to_keep[sousgroupes[36]] = ['codeqvol', 'mdd', 'bio', 'v_4', 'v_8']   
    dict_to_keep[sousgroupes[37]] = ['codeqvol', 'mdd', 'bio', 'v_8', 'v_9']   
    dict_to_keep[sousgroupes[38]] = ['codeqvol', 'mdd', 'v_7']   
    dict_to_keep[sousgroupes[39]] = ['codeqvol', 'mdd', 'bio', 'v_4']   
    dict_to_keep[sousgroupes[40]] = ['codeqvol', 'mdd', 'bio']   
    
    dict_to_keep[sousgroupes[41]] = ['codeqvol', 'mdd', 'bio']   
    dict_to_keep[sousgroupes[42]] = ['codeqvol', 'mdd', 'bio', 'v_4', 'v_5', 'v_7']
    dict_to_keep[sousgroupes[43]] = ['codeqvol', 'mdd', 'bio', 'v_7', 'v_9']   
    dict_to_keep[sousgroupes[44]] = ['codeqvol', 'mdd', 'bio', 'v_4', 'v_10', 'v_12']   
    dict_to_keep[sousgroupes[45]] = ['codeqvol', 'mdd', 'bio', ]   
    dict_to_keep[sousgroupes[46]] = ['codeqvol', 'mdd', 'bio', 'v_5', 'v_7', 'v_17']   
    dict_to_keep[sousgroupes[47]] = ['codeqvol', 'mdd', 'bio', 'v_4', 'v_7']   
    dict_to_keep[sousgroupes[48]] = ['codeqvol', 'mdd', 'bio']   
    dict_to_keep[sousgroupes[49]] = ['codeqvol', 'mdd', 'bio']   
    dict_to_keep[sousgroupes[50]] = ['codeqvol', 'mdd', 'bio']   
    
    dict_to_keep[sousgroupes[51]] = ['codeqvol', 'mdd', 'bio']   
    dict_to_keep[sousgroupes[52]] = ['codeqvol', 'mdd', 'bio', 'v_4']   
    dict_to_keep[sousgroupes[53]] = ['codeqvol', 'mdd', 'bio', 'v_4']   
    dict_to_keep[sousgroupes[54]] = ['codeqvol', 'mdd', 'bio', 'v_3', 'v_5']   
    dict_to_keep[sousgroupes[55]] = ['codeqvol', 'mdd', 'bio']   
    dict_to_keep[sousgroupes[56]] = ['codeqvol', 'mdd', 'bio', ]   
    dict_to_keep[sousgroupes[57]] = ['codeqvol', 'mdd', 'bio']   
    dict_to_keep[sousgroupes[58]] = ['codeqvol', 'mdd', 'bio', 'v_7', 'v_10']   
    dict_to_keep[sousgroupes[59]] = ['codeqvol', 'mdd', 'bio']   
    dict_to_keep[sousgroupes[60]] = ['codeqvol', 'mdd', 'bio', 'v_6']   
    
    dict_to_keep[sousgroupes[61]] = ['codeqvol', 'mdd', 'bio',]   
    dict_to_keep[sousgroupes[62]] = ['codeqvol', 'mdd', 'bio', 'v_4']   
    dict_to_keep[sousgroupes[63]] = ['codeqvol', 'mdd', 'bio', 'v_9']   
    dict_to_keep[sousgroupes[64]] = ['codeqvol', 'mdd', 'bio', 'v_4', 'v_6']   
    dict_to_keep[sousgroupes[65]] = ['codeqvol', 'mdd', 'bio', 'v_6']   
    dict_to_keep[sousgroupes[66]] = ['codeqvol', 'mdd', 'bio']   
    dict_to_keep[sousgroupes[67]] = ['codeqvol', 'mdd', 'bio', 'vn_2', 'v_6']   
    dict_to_keep[sousgroupes[68]] = ['codeqvol', 'mdd', 'bio', 'v_10']   
    dict_to_keep[sousgroupes[69]] = ['codeqvol', 'mdd', 'bio']   
    dict_to_keep[sousgroupes[70]] = ['codeqvol', 'mdd', 'bio', 'v_5']   
    
    dict_to_keep[sousgroupes[71]] = ['codeqvol', 'mdd', 'bio']   
    dict_to_keep[sousgroupes[72]] = ['codeqvol', 'bio', 'v_4', 'v_10']   
    dict_to_keep[sousgroupes[73]] = ['codeqvol', 'mdd', 'bio']   
    dict_to_keep[sousgroupes[74]] = ['codeqvol', 'mdd', 'bio']   
    dict_to_keep[sousgroupes[75]] = ['codeqvol', 'mdd', 'bio', 'vn_2']   
    dict_to_keep[sousgroupes[76]] = ['codeqvol', 'mdd', 'bio']   
    dict_to_keep[sousgroupes[77]] = ['codeqvol', 'mdd', 'bio',]   
    dict_to_keep[sousgroupes[78]] = ['codeqvol', 'mdd', 'bio']   
    dict_to_keep[sousgroupes[79]] = ['codeqvol', 'mdd', 'bio', 'v_4', 'v_6']   
    dict_to_keep[sousgroupes[80]] = ['codeqvol', 'mdd', 'bio', 'v_7']   
    
    dict_to_keep[sousgroupes[81]] = ['codeqvol', 'bio', 'v_8',]  
    dict_to_keep[sousgroupes[82]] = ['codeqvol', 'mdd', 'bio', 'v_3', 'v_10', 'v_24']   
    dict_to_keep[sousgroupes[83]] = ['codeqvol', 'mdd', 'bio',]   
    dict_to_keep[sousgroupes[84]] = ['codeqvol', 'mdd', 'v_9', 'v_10']   
    dict_to_keep[sousgroupes[85]] = ['codeqvol', 'mdd', 'bio',]   
    dict_to_keep[sousgroupes[86]] = ['codeqvol', 'mdd', 'bio', 'v_6', 'v_8']   
    dict_to_keep[sousgroupes[87]] = ['codeqvol', 'mdd',]   
    dict_to_keep[sousgroupes[88]] = ['codeqvol', 'mdd','v_10']   
    dict_to_keep[sousgroupes[89]] = ['codeqvol', 'mdd', 'bio',]    
    dict_to_keep[sousgroupes[90]] = ['codeqvol', 'mdd', 'bio', 'v_4']   
    
    dict_to_keep[sousgroupes[91]] = ['codeqvol', 'bio', 'v_6', 'v_10']   
    dict_to_keep[sousgroupes[92]] = ['codeqvol', 'mdd', 'bio']   
    dict_to_keep[sousgroupes[93]] = ['codeqvol', 'mdd', 'bio', 'v_4', 'v_10']   
    dict_to_keep[sousgroupes[94]] = ['codeqvol', 'bio', 'v_12', 'v_11', 'v_16']   
    dict_to_keep[sousgroupes[95]] = ['codeqvol', 'mdd', 'bio']   
    dict_to_keep[sousgroupes[96]] = ['codeqvol', 'mdd', 'bio', 'v_8']   
    dict_to_keep[sousgroupes[97]] = ['codeqvol', 'mdd', 'bio', 'v_11']   
    dict_to_keep[sousgroupes[98]] = ['codeqvol', 'mdd', 'bio']   
    dict_to_keep[sousgroupes[99]] = ['codeqvol', 'mdd', 'bio', 'v_10']   
    dict_to_keep[sousgroupes[100]] = ['codeqvol', 'mdd', 'bio']   
    
    dict_to_keep[sousgroupes[100]] = ['codeqvol', 'mdd', 'bio']   
    dict_to_keep[sousgroupes[101]] = ['codeqvol', 'bio', 'v_9']   
    dict_to_keep[sousgroupes[102]] = ['codeqvol', 'mdd', 'bio', 'v_5']   
    dict_to_keep[sousgroupes[103]] = ['codeqvol', 'mdd', 'bio', 'v_6', 'v_7']   
    dict_to_keep[sousgroupes[104]] = ['codeqvol', 'mdd', 'bio', 'v_11', 'v_12']   
    dict_to_keep[sousgroupes[105]] = ['codeqvol', 'mdd', 'bio', 'v_5']   
    dict_to_keep[sousgroupes[106]] = ['codeqvol', 'mdd', 'bio']   
    dict_to_keep[sousgroupes[107]] = ['codeqvol', 'mdd', 'bio']   
    dict_to_keep[sousgroupes[108]] = ['codeqvol', 'mdd', 'bio']   
    dict_to_keep[sousgroupes[109]] = ['codeqvol', 'mdd', 'bio', 'v_6', 'v_7']   
    dict_to_keep[sousgroupes[110]] = ['codeqvol', 'mdd', 'bio']   
    
    
    dict_to_keep[sousgroupes[111]] = ['codeqvol', 'bio']   
    dict_to_keep[sousgroupes[112]] = ['codeqvol', 'mdd', 'bio', 'v_9']   
    dict_to_keep[sousgroupes[113]] = ['codeqvol', 'mdd', 'bio', 'v_7', 'v_8']   
    dict_to_keep[sousgroupes[114]] = ['codeqvol', 'mdd', 'bio',]   
    dict_to_keep[sousgroupes[115]] = ['codeqvol', 'mdd', 'bio', 'v_10', 'v_13']   
    dict_to_keep[sousgroupes[116]] = ['codeqvol', 'mdd', 'bio', 'v_7', 'v_9']   
    dict_to_keep[sousgroupes[117]] = ['codeqvol', 'mdd', 'bio']   
    dict_to_keep[sousgroupes[118]] = ['codeqvol', 'mdd', 'bio']   
    dict_to_keep[sousgroupes[119]] = ['codeqvol', 'mdd', 'bio']   
    dict_to_keep[sousgroupes[120]] = ['codeqvol', 'mdd', 'bio']   
    
    dict_to_keep[sousgroupes[121]] = ['codeqvol', 'mdd', 'bio', 'v_8', 'v_11', 'v_12']   
    dict_to_keep[sousgroupes[122]] = ['codeqvol', 'mdd', 'bio']   
    dict_to_keep[sousgroupes[123]] = ['codeqvol', 'mdd', 'bio', 'v_4']   
    dict_to_keep[sousgroupes[124]] = ['codeqvol', 'mdd', 'bio']   
    dict_to_keep[sousgroupes[125]] = ['codeqvol', 'mdd', 'bio', 'v_4', 'v_5']   
    dict_to_keep[sousgroupes[126]] = ['codeqvol', 'bio', 'v_13',]   
    dict_to_keep[sousgroupes[127]] = ['codeqvol', 'mdd', 'bio', 'v_7']   
    dict_to_keep[sousgroupes[128]] = ['codeqvol', 'mdd', 'bio', 'v_6']   
    dict_to_keep[sousgroupes[129]] = ['codeqvol', 'mdd', 'bio', 'v_5']   
    dict_to_keep[sousgroupes[130]] = ['codeqvol', 'mdd', 'bio', 'v_8']   
    
    dict_to_keep[sousgroupes[131]] = ['codeqvol', 'mdd', 'bio', 'v_10']   
    dict_to_keep[sousgroupes[132]] = ['codeqvol', 'mdd', 'bio']   
    dict_to_keep[sousgroupes[133]] = ['codeqvol', 'mdd', 'bio']   
    dict_to_keep[sousgroupes[134]] = ['codeqvol', 'mdd', 'bio', 'v_9']   
    dict_to_keep[sousgroupes[135]] = ['codeqvol', 'mdd', 'bio', 'v_6']   
    dict_to_keep[sousgroupes[136]] = ['codeqvol', 'mdd', 'bio',]   
    dict_to_keep[sousgroupes[137]] = ['codeqvol', 'mdd', 'bio', 'v_6']   
    dict_to_keep[sousgroupes[138]] = ['codeqvol', 'mdd', 'bio', 'v_4']   
    dict_to_keep[sousgroupes[139]] = ['codeqvol', 'mdd', 'bio', 'v_6']   
    dict_to_keep[sousgroupes[140]] = ['codeqvol', 'mdd', 'bio']   
    
    dict_to_keep[sousgroupes[141]] = ['codeqvol', 'mdd', 'bio', 'v_4', 'v_6']   
    dict_to_keep[sousgroupes[142]] = ['codeqvol', 'mdd', 'bio', 'v_9',]   
    dict_to_keep[sousgroupes[143]] = ['codeqvol', 'mdd', 'bio', 'v_8']   
    dict_to_keep[sousgroupes[144]] = ['codeqvol', 'mdd', 'bio', 'v_12',]   
    dict_to_keep[sousgroupes[145]] = ['codeqvol', 'mdd', 'bio', 'v_8']   
    dict_to_keep[sousgroupes[146]] = ['codeqvol', 'mdd', 'bio']   
    dict_to_keep[sousgroupes[147]] = ['codeqvol', 'mdd', 'bio', 'v_4']   
    dict_to_keep[sousgroupes[148]] = ['codeqvol', 'mdd', 'bio', 'v_7']   
    dict_to_keep[sousgroupes[149]] = ['codeqvol', 'mdd', 'bio']   
    dict_to_keep[sousgroupes[150]] = ['codeqvol', 'bio', 'v_13', 'v_20']   
    
    dict_to_keep[sousgroupes[151]] = ['codeqvol', 'mdd', 'bio', 'v_9', 'v_12']   
    dict_to_keep[sousgroupes[152]] = ['codeqvol', 'mdd', 'bio', 'v_19']   
    dict_to_keep[sousgroupes[153]] = ['codeqvol', 'mdd', 'bio', 'v_5']   
    dict_to_keep[sousgroupes[154]] = ['codeqvol', 'mdd', 'bio']   
    dict_to_keep[sousgroupes[155]] = ['codeqvol', 'mdd', 'bio', 'v_5', 'v_11']   
    dict_to_keep[sousgroupes[156]] = ['codeqvol', 'mdd', 'bio', 'v_4', 'v_10']   
    dict_to_keep[sousgroupes[157]] = ['codeqvol', 'mdd', 'bio', 'v_4']   
    dict_to_keep[sousgroupes[158]] = ['codeqvol', 'mdd', 'bio', 'v_8']   
    dict_to_keep[sousgroupes[159]] = ['codeqvol', 'mdd', 'bio']   
    dict_to_keep[sousgroupes[160]] = ['codeqvol', 'mdd', 'bio', 'v_8']   
    
    dict_to_keep[sousgroupes[161]] = ['codeqvol', 'mdd', 'bio', 'v_7']   
    dict_to_keep[sousgroupes[162]] = ['codeqvol', 'mdd', 'bio', 'v_11']   
    dict_to_keep[sousgroupes[163]] = ['codeqvol', 'mdd', 'bio', 'v_7']   
    dict_to_keep[sousgroupes[164]] = ['codeqvol', 'mdd', 'bio']   
    dict_to_keep[sousgroupes[165]] = ['codeqvol', 'mdd', 'bio']   
    dict_to_keep[sousgroupes[166]] = ['codeqvol', 'mdd', 'bio', 'v_6']   
    dict_to_keep[sousgroupes[167]] = ['codeqvol', 'mdd', 'bio', 'v_7']   
    dict_to_keep[sousgroupes[168]] = ['codeqvol', 'mdd', 'bio', 'v_5']   
    dict_to_keep[sousgroupes[169]] = ['codeqvol', 'mdd', 'bio', 'v_8']   
    dict_to_keep[sousgroupes[170]] = ['codeqvol', 'mdd', 'bio', 'v_4']   
    
    dict_to_keep[sousgroupes[171]] = ['codeqvol', 'mdd', 'bio', 'v_8']   
    dict_to_keep[sousgroupes[172]] = ['codeqvol', 'mdd', 'bio']   
    dict_to_keep[sousgroupes[173]] = ['codeqvol', 'mdd', 'bio']   
    dict_to_keep[sousgroupes[174]] = ['codeqvol', 'mdd', 'bio', 'v_6']   
    dict_to_keep[sousgroupes[175]] = ['codeqvol', 'mdd', 'bio', 'v_7']   
    dict_to_keep[sousgroupes[176]] = ['codeqvol', 'mdd', 'bio']   
    dict_to_keep[sousgroupes[177]] = ['codeqvol', 'mdd', 'bio']   
    dict_to_keep[sousgroupes[178]] = ['codeqvol', 'bio', 'v_12']   
    dict_to_keep[sousgroupes[179]] = ['codeqvol', 'mdd', 'bio', 'v_10']   
    dict_to_keep[sousgroupes[180]] = ['codeqvol', 'mdd', 'bio', 'v_9', 'v_11']   
    
    dict_to_keep[sousgroupes[181]] = ['codeqvol', 'mdd', 'bio', 'v_7', 'v_6']  
    dict_to_keep[sousgroupes[182]] = ['codeqvol', 'mdd', 'bio', 'v_5']   
    dict_to_keep[sousgroupes[183]] = ['codeqvol', 'mdd', 'bio', 'v_8', 'v_10']   
    dict_to_keep[sousgroupes[184]] = ['codeqvol', 'mdd', 'bio', 'v_6', 'v_10']   
    dict_to_keep[sousgroupes[185]] = ['codeqvol', 'mdd', 'bio', 'v_6']   
    dict_to_keep[sousgroupes[186]] = ['codeqvol', 'mdd', 'bio', 'v_6']   
    dict_to_keep[sousgroupes[187]] = ['codeqvol', 'mdd', 'bio', 'v_4']   
    dict_to_keep[sousgroupes[188]] = ['codeqvol', 'mdd', 'bio', 'v_4', 'v_13']   
    dict_to_keep[sousgroupes[189]] = ['codeqvol', 'mdd', 'bio']   
    dict_to_keep[sousgroupes[190]] = ['codeqvol', 'mdd', 'bio', 'v_7', 'v_9']   
     
    dict_to_keep[sousgroupes[191]] = ['codeqvol', 'mdd', 'bio']   
    dict_to_keep[sousgroupes[192]] = ['codeqvol', 'mdd', 'bio', 'v_5',]   
    dict_to_keep[sousgroupes[193]] = ['codeqvol', 'mdd', 'bio']   
    dict_to_keep[sousgroupes[194]] = ['codeqvol', 'mdd', 'bio']   
    dict_to_keep[sousgroupes[195]] = ['codeqvol', 'mdd', 'bio', 'v_8']   
    dict_to_keep[sousgroupes[196]] = ['codeqvol', 'mdd', 'bio', 'v_9']   
    dict_to_keep[sousgroupes[197]] = ['codeqvol', 'mdd', 'bio', 'v_5']   
    dict_to_keep[sousgroupes[198]] = ['codeqvol', 'mdd', 'bio', 'v_3', 'v_5']   
    dict_to_keep[sousgroupes[199]] = ['codeqvol', 'mdd', 'bio', 'v_8', 'v_12']   
    dict_to_keep[sousgroupes[200]] = ['codeqvol', 'mdd', 'bio', 'v_4', 'v_6']   
    
    dict_to_keep[sousgroupes[201]] = ['codeqvol', 'mdd', 'bio', 'v_5', 'v_6']   
    dict_to_keep[sousgroupes[202]] = ['codeqvol', 'mdd', 'bio']   
    dict_to_keep[sousgroupes[203]] = ['codeqvol', 'mdd', 'bio', 'v_4', 'v_5', 'v_6']   
    dict_to_keep[sousgroupes[204]] = ['codeqvol', 'mdd', 'bio', 'v_4', 'v_6']
    
    return dict_to_keep