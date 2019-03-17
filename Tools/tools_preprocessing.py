#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 15:32:40 2018

@author: andrei
"""

#outils transformation table simple

import numpy as np
import csv
import pandas as pd
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def delete_max_rows(sparse_matrix, row_names, n):
    sparse_matrix = sparse_matrix.tocsr()
    indexes = np.where(sparse_matrix.max(axis=1).todense() < n)[0]
    
    return sparse_matrix[indexes, :], [row_names[i] for i in indexes]



def delete_max_col(sparse_matrix, col_names, n):
    sparse_matrix = sparse_matrix.tocsc()
    indexes = np.where(sparse_matrix.max(axis=0).todense() < n)[0]
    
    return sparse_matrix[:, indexes], [col_names[i] for i in indexes]

def delete_n_rows(sparse_matrix, row_names, n):
    indexes = np.where(np.sum(sparse_matrix, axis=1) > n)[0]
    
    return sparse_matrix[indexes, :], [row_names[i] for i in indexes]

def delete_rows_entropy(sparse_matrix, row_names, threshold):
    
    sparse_probs = sparse_matrix.multiply(1 / np.sum(sparse_matrix, axis=1))
    sparse_probs = sparse_probs.tocsr()
    log_probs = sparse_probs[:]
    log_probs.data = np.log(log_probs.data)
    entropies = -np.sum(sparse_probs.multiply(log_probs), axis = 1)
    indexes = np.where(entropies > threshold)[0]
    
    return sparse_matrix[indexes, :], [row_names[i] for i in indexes]


def import_week_table(filename_week_table):
    '''
    import data on the weekly activity of households
    
    ARGUMENT:
        filename of the table with the weekly activity of households, i.e.
         how many baskets has the household purchased during that week
        
    RETURN:
        weekly table: table with the household id and the number of weeks active,
         i.e. how many weeks has the household reported one or more baskets
        
        
    '''
    
    week_table = pd.read_csv(filename_week_table)

    week_table['nb_weeks'] = np.sum(week_table[[str(i) for i in range(1,53)]] > 0, axis = 1)
    week_table = week_table[['Unnamed: 0', 'nb_weeks']]

    return week_table



def import_circuit_table(filename_circuit_table):
    '''
    
    import data on which venue the purchases have been made by the households
    
    ARGUMENT:
        filename of the table with the type of venue of purchase, i.e.
         how many baskets have been purchased in each type of venue by the household
        
    RETURN:
        circuit table: table with the household id and the proportion of each type
            of venue in the household shopping habits
        
    '''
    circuit_table = pd.read_csv(filename_circuit_table)
    
    circuit_table_headers = list(circuit_table.columns.values)
    circuit_table_headers.pop(0)
    
    circuit_table['total'] = np.sum(circuit_table[circuit_table_headers], axis = 1)
    circuit_table_headers.pop(0)
    for header in circuit_table_headers:        
        circuit_table[header] /= circuit_table['total']
    circuit_table = circuit_table.drop(columns=['total'])
    
    return circuit_table

def transform_households(datatable, cols_to_remove, cols_median, cols_binarize):
    
    '''
    
    function to transform raw data on households as desired
    
    ARGUMENT:
        datatable : raw datable of households
        cols to remove : columns which do not have interesting information, or too
         many missing values
        cols_median : columns where missing values should be replaced by the median
        cols_binarize : columns to be binarized, i.e. where dummy variables should be
            create
        
        
    RETURN:
        datatable of households where the columns have been processed as intended
        
        
    '''
    
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
    '''
    function to help put data of the households on the same scale

    ARGUMENT:
        datatable : datatable of households
        norm_headers : columns where values should be normalized
        
    RETURN:
        datatable of households where the columns have been normalized

       
    ''' 
    
        
    for header in norm_headers:
        mean = np.mean(np.asarray(datatable[header].dropna()))
        std = np.std(np.asarray(datatable[header].dropna()))
        datatable[header] = (datatable[header] - mean) / std
        

def cap_values(datatable, cap_headers):
    
    '''
    function to help put data of the households on the same scale
    
    ARGUMENT:
        datatable : datatable of households
        cap_headers : columns where values should be caped (here at 3)
        
    RETURN:
        datatable of households where the columns have been caped

       
    ''' 
    for header in cap_headers:
        datatable[header].loc[datatable[header] > 3] = 3



def import_and_transform_households(filename_households):
    '''
    
    function to import raw household information
    
    ARGUMENT:
        filename_households : datatable of raw household information
        
    RETURN:
        datatable of households where the columns have been processed as intended
        and with extra information such as the BMI and units of consumption by house

    ''' 
    
    households = pd.read_csv(filename_households)
    households['bmi'] = households['pds'] / (households['hau'] /100) ** 2 
    households = households.dropna(subset=['bmi'])
    
    nas_col = households.isnull().sum()
    nas_col = list(nas_col.loc[nas_col >0].index.values)

    
    households = transform_households(households,  ['fare', 'hau', 'pds',], nas_col,
                                                       ['etude', 'itra', 'aiur', 'dpts', 'thab', 'scla', 'csp'])

    normalize_columns(households, ['age','rve'])
    cap_values(households, ['mor', 'voit', 'tvc1', 'chie', 'cha', 'en3', 'en6', 'en15', 'en25', 'tlpo'])
    
    households['unite_conso'] = 1 + 0.7 * (households['conjoint'] + households['en25']) + 0.5 * (households['en15'] + households['en6'] + households['en3'])
    
    return households



def merge_households(household_table, week_table, circuit_table):
    
    '''
    function to merge all information about households
    
    ARGUMENT:
        household_table : datatable with processed household information
        week_table : datatable with weekly activity information
        circuit_table : datatable with information about circuit of purchase
        
    RETURN:
        datatable of households where the three datatables have been merged

    ''' 
    
    households = pd.merge(household_table, week_table, left_index = True, right_on = 'Unnamed: 0')
    households = pd.merge(households, circuit_table, left_on = 'Unnamed: 0', right_on = 'Unnamed: 0')
    
    households = households.set_index('Unnamed: 0')
    return households


def load_purchase_matrix(filename_matrix, filename_rows, filename_columns):
    
    '''
    function to load the purchase data
    
    ARGUMENT:
        filename_matrix : name of the sparse matrix file
        filename_rows : name of the rows of the sparse matrix file
        filename_columns : name of the rows of the sparse matrix file
        
    RETURN:
        sparse matrix, name of the rows of the sparse matrix (household id),
         name of the colums of the sparse matrix (products id)

    ''' 
    
    loader = np.load(filename_matrix)
    result = csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])
    with open(filename_rows, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
             row_names = row

    with open(filename_columns, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
             col_names = row
             
    assert len(row_names) == result.shape[0]
    assert len(col_names) == result.shape[1]
        
    return result, row_names, col_names


def map_products(product_table, index = True):
    
    '''
    
    function to help the clustering
    maps each product to one of its attributes, either row index or other
    
    ARGUMENT:
        product_table : name of the raw products table

        
    RETURN:
        dictionary with product key as id, and some value as key

    ''' 
    
    
    result = dict()
    for i, row in product_table.iterrows():
        if index == True:
            
            result[int(row['product'])] = i
        else:
            result[int(row['product'])] = row[index]
            
    return result



def map_product_list(product_list):
    result = dict()
    for i, p in enumerate(product_list):
        result[int(p)] = i
    return result


def create_dict_qvol(products_table):
    
    products_table_subset = products_table[['sousgroupe', 'valqvol', 'count', 'codeqvol']].loc[products_table.reset_index().groupby(['sousgroupe', 'codeqvol'])['count'].idxmax()]
    
    result = dict()
    
    for i, row in products_table_subset.iterrows():
        
        result[row['sousgroupe'] + row['codeqvol']] = row['valqvol']
    
    return result
        
        

def cluster_products(cluster, products_table, purchase_matrix, col_names, verbose = False):
    
    '''
    function to cluster products together, taking into consideration differences 
     in volumes
    
    ARGUMENT:
        cluster : file with cluster of products
        products_table : file with raw info on products
        sparse_matrix : file with the sparse matrix
        
    RETURN:
        sparse matrix with clustered columns taking into account vol differences
        new names of the columns (by cluster representative, i.e. product most purchased in the cluster)
        
    ''' 
    
    
    counter= 0
    col_counter = 0
    values = np.zeros(len(purchase_matrix.data)) 
    column_index = np.zeros(len(purchase_matrix.data )) 
    row_index =np.zeros(len(purchase_matrix.data)) 
    new_col_names = list()
    
    purchase_matrix = purchase_matrix.tocoo().tocsc()

    product_mapping = map_product_list(col_names)
    product_mapping_qvol = map_products(products_table, 'valqvol')
    product_mapping_sousgroupe = map_products(products_table, 'sousgroupe')
    product_mapping_codeqvol = map_products(products_table, 'codeqvol')

    representative_qvol = create_dict_qvol(products_table)
    
    
    
    for i, row in cluster.iterrows():
#        row = cluster.loc[332]
        counter_init = counter
        
        cols = row['1'][1:-1].split(',')
        cols_index = [product_mapping[int(c.rstrip())] for c in cols]
        vol_val = np.reshape(np.asarray([product_mapping_qvol[int(c.rstrip())] for c in cols]), (len(cols), 1))
        
#        test = products_table.loc[cols_index]
        
        vol_val /= representative_qvol[product_mapping_sousgroupe[int(row['0'])] + product_mapping_codeqvol[int(row['0'])]]

        features = np.dot(purchase_matrix[:,cols_index], csr_matrix(vol_val))
#        test2 = features.data
                    
 
        counter_final = counter + len(features.data)
        values[counter_init:counter_final] = features.data
        row_index[counter_init:counter_final] = features.indices
        column_index[counter_init:counter_final] = np.ones(len(features.indices)) * col_counter
        
        
        col_counter += 1
        counter = counter_final
        new_col_names.append(row['0'])
        
        if verbose == True:
            if col_counter % 1000 == 0:
                print(col_counter)



    result = csr_matrix((values[:counter_final], (row_index[:counter_final].astype(int), column_index[:counter_final].astype(int))))
    assert result.shape[1] == len(new_col_names)

    return result, new_col_names



def entropy(data):
    
    ''' 
    function to compute entropy of an array, useful for purging data
    
    ARGUMENT:
        data : array data
        
    RETURN:
        computed entropy     
    ''' 
#    assert np.sum(data) == 1
    assert data.all() >= 0
    data = data[data > 0]
    entropy = -np.dot(data, np.log(data))
    return entropy


def keep_entropy(feature, purge_entropy, threshold = 0):
    ''' 
    function to decide if the data point should be kept or not
    
    ARGUMENT:
        feature : purchase data
        purge_entropy : if to use the entropy or not
        threshold : threshold below which the data point should not be considered
        
        
    RETURN:
        Boolean if the data point should be kept or not
        value of the entropy
    ''' 
    
    en = -1
    if purge_entropy == True:
        try :
            en = entropy(feature.data / np.sum(feature.data))
            if en > threshold:
                to_keep = True
            else:
                to_keep = False
        
        except:
            to_keep = False
    else:
        to_keep = True
    return to_keep, en



def create_value_dictionary(datatable, col_value):
    
    '''
    
    function to help the normalization of the row
    
    ARGUMENT:
        datatable : raw table of households
        col_value : value that should be put as value of the dictionary

        
    RETURN:
        dictionary with household key as id, and some value as key

    ''' 
    result = dict()
    
    for i, row in datatable.iterrows():
        result[i] = row[col_value]
    return result


def normalize_rows_purchase_matrix(purchase_matrix, row_names, households,
                   normalize_purchases = False, binarize_purchases = False,
                   purge_entropy = False, threshold = 2.5):
    
    '''
    
    function to normalize rows according to activity of the household (nb weeks active)
     and size of the household (units of consumption)
    
    ARGUMENT:
        purchase_matrix : file with the purchase matrix
        row_names : file with the names of the rows
        households : file with raw info on households
        
        normalize_purchases : divide each row by the row total
        binarize row : replace all values in the row by 1
        purge_entropy : if some rows should be purged on a entropy criteria
        threshold : threshold for the entropy
        
        
    RETURN:
        sparse matrix with normalized rows having purged or not rows with insufficient info
        new names of the rows, because some rows have been eliminated
        the rejected rows
    ''' 
    
    purchase_matrix = purchase_matrix.tocoo().tocsr()
    
    counter= 0
    row_counter = 0
    values = np.zeros(len(purchase_matrix.data)) 
    column_index = np.zeros(len(purchase_matrix.data )) 
    row_index =np.zeros(len(purchase_matrix.data)) 
    new_row_names = list()
    rejected_rows = list()
    entropy_list = list()
    
    
    dict_weeks = create_value_dictionary(households, 'nb_weeks')
    dict_cons_units = create_value_dictionary(households, 'unite_conso')
    
    for i, row_name in enumerate(row_names):
        
        
        to_keep, en = keep_entropy(purchase_matrix[i,:], purge_entropy, threshold = threshold)   
        entropy_list.append(en)
        if to_keep:
                      
            try:
                features = purchase_matrix[i,:]
                counter_init = counter

                if normalize_purchases == True:
                    data = features.data / np.sum(features.data)
                        
                elif binarize_purchases == True:
                    data = np.ones(len(features.data))
                        
                else:
                    weeks_active = dict_weeks[int(row_name)]
                    purchase_units = dict_cons_units[int(row_name)]
                        
                    data = features.data / (weeks_active * purchase_units)
                    
 
                counter_final = counter + len(features.data)
                values[counter_init:counter_final] = data
                column_index[counter_init:counter_final] = features.indices
                row_index[counter_init:counter_final] = np.ones(len(features.indices)) * row_counter
                        
                row_counter += 1
                counter = counter_final
                new_row_names.append(row_name)
                
            except:
                rejected_rows.append(row_name)


    result = csr_matrix((values[:counter_final], (row_index[:counter_final].astype(int), column_index[:counter_final].astype(int))))
    
    assert len(new_row_names) == result.shape[0]
    
    return result, new_row_names, entropy_list

def trim_sparse_columns(sparse_matrix, col_names, threshold):
    sparse_matrix = sparse_matrix.tocoo()
    rows = np.asarray(sparse_matrix.row)
    cols =np.asarray(sparse_matrix.col)
    sparse_matrix_binarized = csr_matrix((np.ones(len(rows)), (rows.astype(int), cols.astype(int))))
    
    total_products = np.sum(sparse_matrix_binarized, axis = 0)
    indices = np.where(total_products > threshold)[1]
    
    col_names_new = [col_names[i] for i in indices.astype(int)]
    sparse_matrix = sparse_matrix.tocsc()
    sparse_matrix = sparse_matrix[:,indices]
    return sparse_matrix, col_names_new


def log_sparse(sparse_matrix):
    '''
    
    function transform entries in the sparse matrix
    useful technique since the data follows a power low
    
    ARGUMENT:
        sparse_matrix : the sparse_matrix of purchases

        
    RETURN:
        sparse_matrix where the values have been reduced (log 1 + x)
    ''' 
    
    
    values = sparse_matrix.data
    values = np.log(1 + values)
    rows = sparse_matrix.indices
    columns =  sparse_matrix.indptr
    result = csr_matrix((values, rows, columns))
    return result


def csr_setdiag_val(csr, value=0):
    """Set all diagonal nonzero elements
    (elements currently in the sparsity pattern)
    to the given value. Useful to set to 0 mostly.
    """
    if csr.format != "csr":
        raise ValueError('Matrix given must be of CSR format.')
    csr.sort_indices()
    pointer = csr.indptr
    indices = csr.indices
    data = csr.data
    for i in range(min(csr.shape)):
        ind = indices[pointer[i]: pointer[i + 1]]
        j =  ind.searchsorted(i)
        # matrix has only elements up until diagonal (in row i)
        if j == len(ind):
            continue
        j += pointer[i]
        # in case matrix has only elements after diagonal (in row i)
        if indices[j] == i:
            data[j] = value
    csr.eliminate_zeros()
    
    
def plot_by_group(file_name, coordinates, column, original_table, labels, with_legend=True, hot_colors=False):
    original_table_treated = original_table[[column, 'product', 'count']].set_index('product')
#    labels = [str(label) for label in list(labels)]
    original_table_labels = original_table_treated[column].loc[labels]
    original_table_size = original_table_treated['count'].loc[labels]
    
    df = pd.DataFrame(dict(x=coordinates[:,0], y=coordinates[:,1], size = (40 * np.asarray(original_table_size)/np.max(np.asarray(original_table_size)))**2, 
                           label = list(original_table_labels)))
    grouped_labels = sorted(list(set(original_table_labels)))
    # Plot
    plt.figure(figsize=(20,12), dpi=200)    

    fig, ax = plt.subplots()
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    
    if hot_colors == False:
        colors = cm.tab20(np.linspace(0, 1, len(grouped_labels)))
    else:
        colors = cm.YlOrRd(np.linspace(0, 1, len(grouped_labels)))
        
    plt.ylim((-80,80))
    plt.xlim((-80,80))

    
    for i, l in enumerate(grouped_labels):
        
        subset = df.loc[df['label'] == l]
        
        if l == -1:
            ax.scatter(subset['x'], subset['y'], marker='o', label=l, s = subset['size'], c='black', alpha=0.5)
        else:
            
            
            
            if with_legend == True:
                ax.scatter(subset['x'], subset['y'], marker='o', label=l, s = subset['size'], c=colors[i,:], alpha=0.5)
            else:
                ax.scatter(subset['x'], subset['y'], marker='o', s = subset['size'], c=colors[i,:], alpha=0.5)


    if with_legend == True:
        ax.legend(loc='upper right', prop={'size': 7})
    path = 'results_glove3/' + file_name
    plt.savefig(path, dpi=200)



def plot_by_group_simple(file_name, coordinates, column, original_table, labels, with_legend=True, hot_colors=False):

    original_table_labels = original_table[column].loc[labels]
    
    df = pd.DataFrame(dict(x=coordinates[:,0], y=coordinates[:,1], label = list(original_table_labels)))
    grouped_labels = sorted(list(set(original_table_labels)))
    # Plot
    plt.figure(figsize=(20,12), dpi=200)    

    fig, ax = plt.subplots()
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    
    if hot_colors == False:
        colors = cm.tab20(np.linspace(0, 1, len(grouped_labels)))
    else:
        colors = cm.YlOrRd(np.linspace(0, 1, len(grouped_labels)))
        
    plt.ylim((-80,80))
    plt.xlim((-80,80))

    
    for i, l in enumerate(grouped_labels):
        
        subset = df.loc[df['label'] == l]
                   
        if with_legend == True:
            ax.scatter(subset['x'], subset['y'], marker='o', label=l, s = 0.05, c=colors[i,:], alpha=0.5)
        else:
            ax.scatter(subset['x'], subset['y'], marker='o', s = 0.05, c=colors[i,:], alpha=0.5)


    if with_legend == True:
        ax.legend(loc='upper right', prop={'size': 7})
    path = 'results_lsa/' + file_name
    plt.savefig(path, dpi=200)





def get_cluster_representative(cluster, products_table):
    
    prod_count = list()
    for p in cluster:
        
        count = products_table['count'].loc[products_table['product'] == p].values[0]
        prod_count.append(count)
    
    return cluster[np.argmax(prod_count)]
    
    
def cluster_representation(cluster_list, prod_names, prod_table):
    assert len(cluster_list) == len(prod_names)
    result = dict()
    for i, j in enumerate(cluster_list):
        try:
            result[j].append(prod_names[i])
        except:
            result[j] = list()
            result[j].append(prod_names[i])
            
    result_final = list()
    for i in result.keys():
        result_final.append((get_cluster_representative(result[i], prod_table), result[i]))
    
    result_final = pd.DataFrame(result_final)
    result_final = result_final.rename(columns={0:'0', 1:'1'})
    result_final['1'] = result_final['1'].astype(str)
    return result_final


    
    
    