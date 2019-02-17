#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 09:46:15 2018

@author: andrei
"""

#regression tools

import numpy as np
from scipy.stats import pearsonr
from  sklearn.metrics import accuracy_score, mean_squared_error, f1_score, precision_score, recall_score, roc_auc_score
import csv
import pandas as pd
from scipy.sparse import csr_matrix, vstack
from sklearn.cluster import KMeans
import math


def import_week_table():
    week_table = pd.read_csv('data_cleaned/household_activity_week.csv')

    week_table['nb_weeks'] = np.sum(week_table[[str(i) for i in range(1,53)]] > 0, axis = 1)
    week_table = week_table[['Unnamed: 0', 'nb_weeks']]

    return week_table

def import_circuit_table():
    circuit_table = pd.read_csv('data_cleaned/household_activity_circuit.csv')
    
    circuit_table_headers = list(circuit_table.columns.values)
    circuit_table_headers.pop(0)
    
    circuit_table['total'] = np.sum(circuit_table[circuit_table_headers], axis = 1)
    circuit_table_headers.pop(0)
    for header in circuit_table_headers:        
        circuit_table[header] /= circuit_table['total']
    circuit_table = circuit_table.drop(columns=['total'])
    
    return circuit_table


def import_and_transform_households():

    households = pd.read_csv('data_cleaned/foyers_traites.csv')
    households['bmi'] = households['pds'] / (households['hau'] /100) ** 2 
    households = households.dropna(subset=['bmi'])
    
    nas_col = households.isnull().sum()
    nas_col = list(nas_col.loc[nas_col >0].index.values)

    
    households = transform_households(households,  ['fare', 'hau', 'pds',], nas_col,
                                                       ['etude', 'itra', 'aiur', 'dpts', 'thab', 'scla', 'csp'])

    normalize_columns(households, ['age','rve'])
    cap_values(households, ['mor', 'voit', 'tvc1', 'chie', 'cha', 'en3', 'en6', 'en15', 'en25', 'tlpo'])
    
    return households


def merge_households():
    week_table = import_week_table()
    circuit_table = import_circuit_table()
    household_table = import_and_transform_households()
    
    households = pd.merge(household_table, week_table, left_index = True, right_on = 'Unnamed: 0')
    households = pd.merge(households, circuit_table, left_on = 'Unnamed: 0', right_on = 'Unnamed: 0')
    
    households = households.set_index('Unnamed: 0')
    return households




def import_products():
    with open('data/products_matrix.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
             liste_products = row
    
    products_table = pd.read_csv('data/produits_achats.csv')
    products_table = products_table.loc[products_table['product'].isin(liste_products)]
    products_table = clean_table(products_table)
    
    return liste_products, products_table


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])

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

def custom_cross_validation(model, features, target, folds = 5, classification = False):
    np.random.seed(123)
    indexes = np.random.randint(0, folds, len(target))
    
    results_train = list()
    results_test = list()
    
    for i in range(folds):
        index = np.where(indexes == i)[0]
        index_rest = np.where(indexes != i)[0]
        train_x = features[index_rest, :]
        test_x = features[index, :]
        train_y = target[index_rest]
        test_y = target[index]
        
        model.fit(train_x, train_y)
        
        pred_train = model.predict(train_x)
        pred_test = model.predict(test_x)
        
        if classification ==  False:
            result_train = [model.score(train_x, train_y), mean_squared_error(train_y,pred_train)]
            result_test = [model.score(test_x, test_y),  mean_squared_error(test_y, pred_test)]
        else :
#            score_train = model.predict_proba(train_x)[:,1]
#            score_test = model.predict_proba(test_x)[:,1]
            result_train = [recall_score(train_y, pred_train), precision_score(train_y, pred_train), f1_score(train_y, pred_train)]
            result_test = [recall_score(test_y, pred_test), precision_score(test_y, pred_test), f1_score(test_y, pred_test)]
        
        results_train.append(result_train)
        results_test.append(result_test)
        
        print(i)
        
    return results_train, results_test

def one_vs_rest_classifier(model, features, target):

    regression_target = list(set(target))
    results = dict()
    
    for t in regression_target:
        
        indexes_true = np.where(target ==  t)[0]
        targets_new = np.zeros(len(target))
        targets_new[indexes_true] = 1
        
        sc_train, sc_test = custom_cross_validation(model, features, targets_new, classification = True)
        
        results[t] = [np.sum(targets_new), np.mean(sc_train), np.mean(sc_test)]
    
    return results


def transform_summary_regression(summary, variables_list):
    for i, var in enumerate(variables_list):
        summary = summary.replace('x' + str(i+1) + ' ', var + ' ')
    return summary


def get_description_products(products_clusters, dict_description, products_table):
    result = list()
    for i, row in products_clusters.iterrows():
        
        row_bis = products_table.loc[products_table['product'] == row['0']].reset_index().loc[0]
        description = list(row_bis[dict_description[row_bis['sousgroupe']]])
        description.insert(0, row_bis['sousgroupe'])
        description = [str(d) for d in description]
        result.append(';'.join(description))
    return result


def create_simple_mapping(mapping_list):
    result = dict()
    for i, value in enumerate(mapping_list):
        result[value] = i
    return result


def create_clusters_lda(lda_table, list_products, nclusters):
    kmeans = KMeans(n_clusters = nclusters)
    model = kmeans.fit(lda_table)
    clusters = model.labels_
    result = dict()
    for i, cl in enumerate(clusters):
        try:
            result[cl].append(i)
        except:
            result[cl] = list()
            result[cl].append(i)
    return result


def create_sousgroupe_clusters(products_table, sousgroupes, list_products):
    mapping_clusters = create_simple_mapping(sousgroupes)
    mapping_products = create_simple_mapping(list_products)
    result = dict()
    for i, row in products_table.iterrows():
        key = mapping_clusters[row['sousgroupe']]

        try:
            result[key].append(mapping_products[str(row['product'])])
        except:
            result[key] = list()
            result[key].append(mapping_products[str(row['product'])])
    return result            



def transform_households(datatable, cols_to_remove, cols_median, cols_binarize):
    cols_features = list(datatable.columns.values)

    for col in cols_to_remove:
        cols_features.remove(col)
    
    dict_median = dict()
    for col in cols_median:
        try:
            dict_median[col] = np.median(datatable[col].dropna())
        except:
            pass
    
    datatable = datatable.fillna(dict_median)
    datatable = datatable[cols_features]
    datatable = pd.get_dummies(datatable, columns=cols_binarize, drop_first=True)
    datatable = datatable.dropna()
    datatable = datatable.set_index('household')
    return datatable

       
        
def normalize_columns(datatable, norm_headers):
    for header in norm_headers:
        mean = np.mean(np.asarray(datatable[header].dropna()))
        std = np.std(np.asarray(datatable[header].dropna()))
        datatable[header] = (datatable[header] - mean) / std
        

def cap_values(datatable, cap_headers):
    for header in cap_headers:
        datatable[header].loc[datatable[header] > 3] = 3

        
def create_dictionary_households(list_households):
    result = dict()
    for i,h in enumerate(list_households):
        result[h] = i
    return result



def discretize(value, limit_extreme):
    if value < 17.5:
        result = 0
    elif value >= 17.5 and value < 25:
        result = 0
    elif value >= 25 and value < 30 and limit_extreme:
        result = 0
    elif value >= 25 and value < 30 :
        result = 1
    elif value >= 30:
        result = 1
    return result
      

def is_worst_bmi(correlation_bmi, index, sexe, predict_worse):
    if predict_worse == True:
        try :
            value_bmi = correlation_bmi.loc[index]
            if value_bmi['bmi_x'] > value_bmi['bmi_y']:
                worse_bmi = 1
            else:
                worse_bmi = 0
            worse_bmi_final = (worse_bmi == sexe)
        
        except:
            worse_bmi_final = 'single'
    else:
        worse_bmi_final = True
    return worse_bmi_final



def clustered_representations(sparse_matrix, clusters):
    result = np.zeros((sparse_matrix.shape[0], len(clusters)))
    for i in range(len(clusters)):
        value = np.sum(sparse_matrix[:, clusters[i]], axis = 1)
        result[:,i] = np.ravel(value)
    result = np.log(1+result)
    return result



def create_features_table(achats_households, list_households, households,
                          correlation_bmi, lda_households = None,
                          mode="lda", 
                          discretize_target = False,  
                          predict_worse = False, include_socio = False,
                          limit_extreme = False):
    
    regression_features = list()
    regression_target = list()
    regression_id = list()
    dict_index_households = create_dictionary_households(list_households)
    rejected_rows = list()
    sample_weights = list()
        
    if mode == "raw":
        counter= 0
        nb_col = achats_households.shape[1]
        row_counter = 0
        values = np.zeros(len(achats_households.data) * 3) 
        column_index = np.zeros(len(achats_households.data ) * 3) 
        row_index =np.zeros(len(achats_households.data ) *3) 
    for i, row in households.iterrows():
        
        
        is_worst = is_worst_bmi(correlation_bmi, i, int(row['sexe']), predict_worse)        
        
        if (not math.isnan(row['bmi'])) and is_worst:
            
            
            if discretize_target == False:
                target = row['bmi']
            else:
                target = discretize(row['bmi'], limit_extreme)

            h = str(int(i))
            sexe = row['sexe']
            
            try:
                if mode == 'raw':
                    features = achats_households[dict_index_households[h],:]
                    counter_init = counter

                    data = features.data

                    
                    if include_socio == True:
                        row = list(row.drop(['bmi']))
                        counter_final = counter + len(features.data) + len(row)


                        values[counter_init:counter_final] = np.append(data, np.asarray(row))
                        column_index[counter_init:counter_final] = np.append(features.indices, np.asarray(range(nb_col,nb_col + len(row))))
                        row_index[counter_init:counter_final] = np.ones(len(features.indices) + len(row)) * row_counter
                        

                    else:
                        counter_final = counter + len(features.data)
                        values[counter_init:counter_final] = data
                        column_index[counter_init:counter_final] = features.indices
                        row_index[counter_init:counter_final] = np.ones(len(features.indices)) * row_counter
 
                    if is_worst == 'single':
                        sample_weights.append(1)
                    else:
                        sample_weights.append(0.5)
                        
                    row_counter += 1
                    counter = counter_final
                
                elif mode == 'lda':
                    if include_socio == True:
                        row = list(row.drop(['bmi']))
                        regression_features.append(np.append(lda_households[dict_index_households[h],:], np.asarray(row)))
                    else:
                        regression_features.append(lda_households[dict_index_households[h],:])                        
                
                elif mode == 'socio-eco':
                    row = row.drop(['bmi'])
                    regression_features.append(list(row))
                
                regression_id.append(h + str(int(sexe)))
                regression_target.append(target)
            
            except:
                rejected_rows.append(row)

    if (mode == 'lda') or (mode == 'socio-eco') or ((mode == 'clusters')):
        result = np.asarray(regression_features)
    elif mode == 'raw':
        result = csr_matrix((values[:counter_final], (row_index[:counter_final].astype(int), column_index[:counter_final].astype(int))))
    
    
    regression_target = np.asarray(regression_target)
    return result, regression_target , regression_id, sample_weights






def log_sparse(sparse_matrix):
    values = sparse_matrix.data
    values = np.log(1 + values)
    rows = sparse_matrix.indices
    columns =  sparse_matrix.indptr
    result = csr_matrix((values, rows, columns))
    return result



def entropy(data):
#    assert np.sum(data) == 1
    assert data.all() >= 0
    data = data[data > 0]
    entropy = -np.dot(data, np.log(data))
    return entropy

def map_products(product_table, index = True):
    result = dict()
    for i, row in product_table.iterrows():
        if index == True:
            
            result[int(row['product'])] = i
        else:
            result[int(row['product'])] = row[index]
            
    return result



def cluster_products(cluster, products_table, sparse_matrix):
    sparse_matrix = sparse_matrix.tocoo().tocsc()
    result = np.zeros((sparse_matrix.shape[0], len(cluster)))
    
    product_mapping = map_products(products_table)
    product_mapping_qvol = map_products(products_table, 'valqvol')
       
    for i, row in cluster.iterrows():
        cols = row['1'][1:-1].split(',')
        cols_index = [product_mapping[int(c.rstrip())] for c in cols]
        vol_val = np.reshape(np.asarray([product_mapping_qvol[int(c.rstrip())] for c in cols]), (len(cols), 1))
        vol_val /= product_mapping_qvol[int(row['0'])]

        result[:,i] = np.ravel(np.dot(sparse_matrix[:,cols_index], csr_matrix(vol_val)).todense())
        
    result = csr_matrix(result)
    return result


    
def vizalisation_coefficients(products_names, coeficients, products_table, quartiles):
    
    data = {'product': products_names, 'coef' : coeficients}
    coef_dataframe = pd.DataFrame(data=data)
    
    products = products_table[['product', 'sousgroupe']]
    final_dataframe = pd.merge(coef_dataframe, products, left_on = 'product', right_on = 'product')
    final_dataframe['percentile'] = pd.qcut(final_dataframe['coef'], quartiles, labels=False, duplicates = 'drop')
    
    sousgroupes = sorted(list(set(final_dataframe['sousgroupe'])))
    result = np.zeros((len(sousgroupes), quartiles ))
    
    for i, sousgroupe in enumerate(sousgroupes):
        
        subset = final_dataframe.loc[final_dataframe['sousgroupe']== sousgroupe]
        subset_groupes = subset[['sousgroupe','percentile']].groupby('percentile').count()
        subset_groupes['sousgroupe'] /= len(subset)
        
        for j, row in subset_groupes.iterrows():
            result[i, j] = row['sousgroupe']
        
    result = pd.DataFrame(result, index = sousgroupes, columns=range(quartiles))
    return result


def determine_category(type_categorization, attribute1, attribute2 = ''):
    
    assert type_categorization in ['gender', 'marital', 'gender_marital', 'all']
    
    if type_categorization == 'gender':
        if attribute1 == 0:
            cat = 1
        elif attribute1 == 1:
            cat = 2
    
    if type_categorization == 'marital':
        if attribute2 == 0:
            cat = 1
        elif attribute2 == 1:
            cat = 2

    if type_categorization == 'gender_marital':

        if (attribute1 == 0) and (attribute2 == 0):
            cat = 1
        elif (attribute1 == 0) and (attribute2 == 1):
            cat = 2    
        elif attribute1 == 1 :
            cat = 3
            
    if type_categorization == 'all':

        if (attribute1 == 0) and (attribute2 == 0):
            cat = 1
        elif (attribute1 == 0) and (attribute2 == 1):
            cat = 2    
        elif (attribute1 == 1) and (attribute2 == 0) :
            cat = 3
        elif (attribute1 == 1) and (attribute2 == 1) :
            cat = 4
    
    return cat




def create_features_table_da(achats_households, list_households, households, type_category,
                          include_socio = False):
    achats_households = achats_households.tocsr()
    regression_target = list()
    regression_id = list()
    dict_index_households = create_dictionary_households(list_households)
    rejected_rows = list()

    counter= 0
    nb_col = achats_households.shape[1]
    row_counter = 0
    values = np.zeros(len(achats_households.data) * 10) 
    column_index = np.zeros(len(achats_households.data ) * 10) 
    row_index =np.zeros(len(achats_households.data ) *10) 
        
        
    for i, row in households.iterrows():
        
        
        if (not math.isnan(row['bmi'])):
            
            target = row['bmi']
            h = str(int(i))
            sexe = row['sexe']
            couple = row['conjoint']            

            cat = determine_category(type_category, sexe, couple)
            
            try:
                
                features = achats_households[dict_index_households[h],:]

                    
                if include_socio == True:
                    row = list(row.drop(['bmi']))
                    span_features = len(features.data) + len(row)
                    counter_final = counter + span_features

                    data_values = np.append(features.data, np.asarray(row))
                    col_values = np.append(features.indices, np.asarray(range(nb_col,nb_col + len(row))))
                    values[counter:counter_final] = data_values
                    column_index[counter:counter_final] = col_values
                    row_index[counter:counter_final] = np.ones(len(features.indices) + len(row)) * row_counter
                        

                else:
                    span_features = len(features.data)
                    counter_final = counter + span_features
                        
                    data_values = features.data
                    col_values = features.indices
                        
                    values[counter:counter_final] = data_values
                    column_index[counter:counter_final] = col_values
                    row_index[counter:counter_final] = np.ones(len(features.indices)) * row_counter

                counter = counter_final
                counter_final = counter + span_features
                values[counter:counter_final] = data_values
                column_index[counter:counter_final] = col_values + (cat * nb_col)
                row_index[counter:counter_final] = np.ones(len(features.indices)) * row_counter                    
                
                
                regression_id.append(h + str(int(sexe)))
                regression_target.append(target)                    

                row_counter += 1
                counter = counter_final
            
            except:
                rejected_rows.append(row)

    result = csr_matrix((values[:counter_final], (row_index[:counter_final].astype(int), column_index[:counter_final].astype(int))))
    regression_target = np.asarray(regression_target)
    
    
    return result, regression_target , regression_id

