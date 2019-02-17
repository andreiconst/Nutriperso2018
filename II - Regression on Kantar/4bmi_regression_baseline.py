#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 10:13:35 2018

@author: andrei
"""

#baseline bmi regression

# =============================================================================
# Import libraries and relevant data
# =============================================================================
import pandas as pd
import numpy as np
import csv
import statsmodels.api as sm
from sklearn import linear_model
import sys
sys.path.insert(0, '/home/andrei/Desktop/kantar_2014/code_produits/')
sys.path.insert(0, '/home/andrei/Desktop/kantar_2014/code_foyers/')
from tools_regression import *
from foyers_processing import *
from foyers_clustering_tools import *
from sklearn import tree


households = pd.read_csv('data/foyers_traites.csv')
households['bmi'] = households['pds'] / (households['hau'] /100) ** 2 

labels, transformed_households_pca = read_table_and_clean('data/transformed_householdsPCA.csv', 21)
_, transformed_households_ae = read_table_and_clean('data/transformed_households_antoencoder.csv', 21)

# =============================================================================
# Put data into good format
# =============================================================================

def create_features_transformed(labels, datatable_features, datatable_bmi):
    labels = labels.astype(str)
    dataframe_features = pd.DataFrame(datatable_features, index = labels)
    datatable_bmi['sexe'] = datatable_bmi['sexe'].astype(str) 
    datatable_bmi['household'] = datatable_bmi['household'].astype(str) 
    datatable_bmi['code'] = datatable_bmi['household'] + datatable_bmi['sexe']
    datatable_bmi_pruned = datatable_bmi[['code','bmi']]
    datatable_final = pd.merge(dataframe_features, datatable_bmi_pruned, left_index=True, right_on = 'code', how='inner')
    datatable_final = datatable_final.drop(columns=['code'])
    datatable_final = datatable_final.dropna()
    target =  np.asarray(list(datatable_final['bmi']))
    features =  np.asarray(datatable_final.drop(columns = ['bmi']))
    return features, target

features_pca, target_pca = create_features_transformed(labels, transformed_households_pca, households)
features_ae, target_ae = create_features_transformed(labels, transformed_households_ae, households)

np.mean(target_ae)

regr = linear_model.LinearRegression()
scores_train, scores_test = custom_cross_validation(regr, features_pca, target_pca)
print(np.mean([scores_train[i][0] for i in range(len(scores_train))]))
print(np.mean([scores_train[i][1] for i in range(len(scores_train))]))

print(np.mean([scores_test[i][0] for i in range(len(scores_test))]))
print(np.mean([scores_test[i][1] for i in range(len(scores_test))]))

print('------------------------------------------------------------')


# =============================================================================
# Run regression normal
# =============================================================================

households = households.dropna(subset=['bmi'])
nas_col = households.isnull().sum()
nas_col = list(nas_col.loc[nas_col >0].index.values)


def create_feature_target(datatable, target_label, cols_to_remove, cols_median, cols_binarize):
    cols_features = list(datatable.columns.values)
    cols_features.remove(target_label)
    for col in cols_to_remove:
        cols_features.remove(col)
    
    dict_median = dict()
    for col in cols_median:
        try:
            dict_median[col] = np.median(datatable[col].dropna())
        except:
            pass
    
    datatable = datatable.fillna(dict_median)
    datatable_features = datatable[cols_features]
    datatable_features = pd.get_dummies(datatable_features, columns=cols_binarize, drop_first=True)
    datatable_features[target_label] = datatable[target_label]
    datatable_features = datatable_features.dropna()
    target = np.asarray(list(datatable_features[target_label]))
    datatable_features = datatable_features.drop(columns=[target_label])
    var_names = list(datatable_features.columns.values)
    features = np.asarray(datatable_features)

    
    return features, target, var_names

features_normal, target_normal, var_normal = create_feature_target(households, 'bmi', ['fare','household', 'hau', 'pds',], nas_col,
                                                       ['etude', 'itra', 'aiur', 'dpts', 'thab', 'scla', 'csp'])


#regr = linear_model.LinearRegression()
regr = tree.DecisionTreeRegressor(max_features = 90, max_depth = 5)
scores_train, scores_test = custom_cross_validation(regr, features_normal, target_normal)
print(np.mean([scores_train[i][0] for i in range(len(scores_train))]))
print(np.mean([scores_train[i][1] for i in range(len(scores_train))]))
print(np.mean([scores_test[i][0] for i in range(len(scores_test))]))
print(np.mean([scores_test[i][1] for i in range(len(scores_test))]))
print('------------------------------------------------------------')

#Print results normal

X2 = sm.add_constant(features_normal)
est = sm.OLS(target_normal, X2)
est2 = est.fit()
print(est2.summary())
summary = str(est2.summary())
test = transform_summary_regression(summary, var_normal)
with open("regression_results/regression_baseline.txt", "w") as f:
    f.write(test)
    
# =============================================================================
# Run regression clusters
# =============================================================================

clustering_ae = pd.read_csv('data/clusters_agglomerative_ae12.csv', header=None)    
clustering_pca = pd.read_csv('data/clusters_agglomerative_pca12.csv', header=None)    


def create_features_clustering(datatable_clustering, datatable_bmi):
    datatable_bmi['sexe'] = datatable_bmi['sexe'].astype(str) 
    datatable_bmi['household'] = datatable_bmi['household'].astype(str) 
    datatable_bmi['code'] = datatable_bmi['household'] + datatable_bmi['sexe']
    datatable_clustering[0] = datatable_clustering[0].astype(str)
    datatable_final = pd.merge(datatable_bmi, datatable_clustering, left_on = 'code', right_on = 0)
    datatable_final = datatable_final.dropna(subset=['bmi'])
    #datatable_features = datatable_final[1]
    #datatable_features = pd.get_dummies(datatable_features, columns=[1], drop_first=True)
    #features = np.asarray(datatable_features)
    #target = np.asarray(datatable_final['bmi'])
    datatable_final = datatable_final[[1, 'bmi']]
    return datatable_final

features_cl_pca = create_features_clustering(clustering_pca, households)
summary = features_cl_pca.groupby([1])['bmi'].mean()
summary_std = features_cl_pca.groupby([1])['bmi'].std()

list_values = list()
for i in range(12):
    list_values.append('%.2f +- %.2f' % (summary.loc[i], summary_std.loc[i]))

regr = linear_model.LinearRegression()
scores_train, scores_test = custom_cross_validation(regr, features_cl_ae, target_cl_ae)
print(np.mean([scores_train[i][0] for i in range(len(scores_train))]))
print(np.mean([scores_train[i][1] for i in range(len(scores_train))]))

print(np.mean([scores_test[i][0] for i in range(len(scores_test))]))
print(np.mean([scores_test[i][1] for i in range(len(scores_test))]))

print('------------------------------------------------------------')


#Print results readable format

cluster_names_correspondance = ['young professionals', 'middle class families', 'lower class youth',
                                'upper class mid-age no children', 'upper class young families', 'working youth',
                                'lower class mature family', 'upper class retirees', 'middle class retirees',
                                'upper class families', 'countryside family', 'lower class young families']

X2 = sm.add_constant(features_cl_pca)
est = sm.OLS(target_cl_pca, X2)
est2 = est.fit()
print(est2.summary())
summary = str(est2.summary())
test = transform_summary_regression(summary, cluster_names_correspondance[1:])
with open("regression_results/regression_clustering_baseline.txt", "w") as f:
    f.write(test)
