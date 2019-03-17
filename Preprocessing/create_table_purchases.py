#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 14:57:34 2018

@author: andrei
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix




products_table = pd.read_csv('data/produits_achats.csv', encoding='latin1')
full_table = pd.read_csv('data/achat_2014cleaned.csv', encoding='latin1')

def create_dict_table(datatable, header):
    result = dict()
    for i, row in datatable.iterrows():
        result[row[header]] = i
    return result

def create_dict_list(list1):
    result = dict()
    for i,l in enumerate(list1):
        result[l] = i
    return result





def transform_aggregated_datatable(datatable, var1, var2, val, sparse = False):

    if type(var1) == type(list()):
        headers_all = var1[:]
        headers_all.extend([var2, val])
        datatable = datatable[headers_all]
        datatable['new_var1'] = ''

        for var in var1:
            datatable['new_var1'] += datatable[var].astype(str) + ';'
        var1 = 'new_var1'
        
    list_var1 = sorted(list(datatable[var1].drop_duplicates()))
    list_var2 = sorted(list(datatable[var2].drop_duplicates()))

    dict_index_var1 = create_dict_list(list_var1)
    dict_index_var2 = create_dict_list(list_var2)
    
    if sparse == False:
        table = datatable[[var1, var2, val]].drop_duplicates()
        table_final = table.groupby([var1, var2]).count().reset_index()
        result = np.zeros((len(list_var1), len(list_var2)))

    else:
        table_final = datatable[[var1, var2, val]].groupby([var1, var2]).sum().reset_index()        
        cols = np.zeros(len(table_final))
        rows = np.zeros(len(table_final))
        values = np.zeros(len(table_final))
        

    counter = 0
    for tup, row in table_final.iterrows():
        
        if sparse == False:
            i = dict_index_var1[row[var1]]
            j = dict_index_var2[row[var2]]
            
            result[i,j] = row[val]
        
        else:
            rows[counter] = dict_index_var1[row[var1]]
            cols[counter] = dict_index_var2[row[var2]]
            values[counter] = row[val]
            counter += 1
            
            if counter % 100000 == 0:
                print(counter)
            
    if sparse == False:
        result_final = pd.DataFrame(result, index = list_var1, columns= list_var2)
        return result_final
    else:
        result_final = csr_matrix((values, (rows.astype(int), cols.astype(int))))
        return result_final, list_var1, list_var2

        
datatable_purchases, hh, products = transform_aggregated_datatable(full_table, 'household', 'product', 'qaachat', sparse = True)
datatable_purchases_weekly, hh_and_weeks, products_week = transform_aggregated_datatable(full_table, ['household', 'semaine'], 'product', 'qaachat', sparse = True)
datatable_purchases_month, hh_and_month, products_month = transform_aggregated_datatable(full_table, ['household', ':eriode'], 'product', 'qaachat', sparse = True)
datatable_purchases_cp, hh_and_cp, _ = transform_aggregated_datatable(full_table, ['household', 'codepanier'], 'product', 'qaachat', sparse = True)




def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

def save_purchase_matrix(filename_matrix, sparse_matrix, filename_rows, rows_list):
    

    save_sparse_csr('data/' + filename_matrix, sparse_matrix)


    hh = [str(h) for h in rows_list]
    
    with open('data/' + filename_rows,'w') as f:
        f.write(','.join(hh))
    


save_purchase_matrix('purchase_table_full', datatable_purchases, 'households_matrix_full.txt', hh, 'products_matrix_full.txt', products)
save_purchase_matrix('purchase_table_full_weekly', datatable_purchases_weekly, 'households_week_matrix_full.txt', hh_and_weeks, 'products_matrix_full_weekly.txt', products)
save_purchase_matrix('purchase_table_full_monthly', datatable_purchases_month, 'households_month_matrix_full.txt', hh_and_month, 'products_matrix_full_month.txt', products_month)
save_purchase_matrix('purchase_table_codepanier', datatable_purchases_cp, 'households_codepanier.txt', hh_and_cp)


# =============================================================================
# Run some tests, do not pay attention
# =============================================================================
'''
# Same distribution ??

products.index('154')
product1 = 97
product2 = 67927

distr1 = np.ravel(datatable_purchases[:,product1][datatable_purchases[:,product1] > 0])
distr2 = np.ravel(datatable_purchases[:,product2][datatable_purchases[:,product2] > 0])

from scipy.stats import ks_2samp

test1 = ks_2samp(distr1, distr2)


# Vector product ?

product1 = 11910
product2 = 97


norm1 = np.sqrt(np.dot(datatable_purchases[:,product1].T, datatable_purchases[:,product1])[0,0])
norm2 = np.sqrt(np.dot(datatable_purchases[:,product2].T, datatable_purchases[:,product2])[0,0])

dot_product = np.dot(datatable_purchases[:,product1].T, datatable_purchases[:,product2])[0,0] / (norm1 * norm2)
'''