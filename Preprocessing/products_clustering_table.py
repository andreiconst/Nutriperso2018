#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Apr  3 17:15:24 2018

@author: andrei
"""

import pandas as pd
import numpy as np
import string
printable = set(string.printable)
import matplotlib.pyplot as  plt
import sys
import os
sys.path.insert(0, os.getcwd() + '/Preprocessing')
from tools_clustering import *


# =============================================================================
# Run clustering mano
# =============================================================================


products_table = pd.read_csv(os.getcwd() + '/data/produits_achats.csv')
dict_qval = create_dict_val_qvol(products_table)
products_table = clean_table(products_table)


products_table = products_table.drop(columns=['valqvol'])
sousgroupes = sorted(list(set(products_table['sousgroupe'])))

dict_to_keep = create_dict_to_keep(sousgroupes)
clusters = clustering_algo_full_mano(products_table, sousgroupes, dict_to_keep,dict_qval)


clustered_datatable = create_pruned_datatable(products_table, clusters)


# =============================================================================
# Run automatic clusteing
# =============================================================================

'''
cols_to_drop = ['fabricant', 'marque'] # 

clusters1 = clustering_algo_simple_all(products_table, sousgroupes, cols_to_drop, dict_qval)  
result1 = count_clusters(clusters1)

clusters2 = clustering_algo_simple_all(products_table, sousgroupes, cols_to_drop, dict_qval, cluster_dict = clusters1, advanced = True)  
result2 = count_clusters(clusters2)  
              
merged_clusters = merge_clusters(products_table, clusters1, clusters2)
clustered_datatable = create_pruned_datatable(products_table, merged_clusters)
'''


# =============================================================================
# Last checks and prints
# =============================================================================
assert np.sum(products_table['count']) == compute_total_count(clustered_datatable)


pruned_products = pruned_clusters(clustered_datatable, 10)

final_dataframe = create_dataframe(pruned_products)
final_dataframe.to_csv('data/cluster_products_mano.csv', index = False)







'''
groupe_info1 = products_table[['groupe', 'sousgroupe']].loc[products_table['product'].isin(pruned_products.keys())].groupby('groupe').count()
groupe_info2 = products_table[['groupe', 'count']].loc[products_table['product'].isin(pruned_products.keys())].groupby('groupe').sum()
groupe_info = pd.merge(groupe_info1, groupe_info2, left_index=True, right_index=True)
groupe_info.to_csv('info_groupes_cluster_mano.csv')


sousgroupe_info1 = products_table[['groupe', 'sousgroupe']].loc[products_table['product'].isin(pruned_products.keys())].groupby('sousgroupe').count()
sousgroupe_info2 = products_table[['sousgroupe', 'count']].loc[products_table['product'].isin(pruned_products.keys())].groupby('sousgroupe').sum()
sousgroupe_info = pd.merge(sousgroupe_info1, sousgroupe_info2, left_index=True, right_index=True)
sousgroupe_info.to_csv('info_sousgroupes_cluster_mano.csv')

'''





# =============================================================================
# Table subset chosen features
# =============================================================================
'''

table_variables = list()
for key in dict_to_keep:
    temp = list()
    for l in dict_to_keep[key]:
        if (l[:2] == 'v_'):
            new_l = 'c_' + l[2:]
            value = products_table[new_l].loc[products_table['sousgroupe'] == key].reset_index().loc[0].values[1]
            temp.append(value)
        elif (l[:3] == 'vn_'):
            new_l = 'cn_' + l[3:]
            value = products_table[new_l].loc[products_table['sousgroupe'] == key].reset_index().loc[0].values[1]
            temp.append(value)  
        else:
            temp.append(l)
    table_variables.append([key, temp])

table_variables = pd.DataFrame(table_variables)
table_variables.to_csv('table_variables_clustering_mano.csv')


'''

# =============================================================================
# Plot 1 : consumption habits ppl
# =============================================================================
'''
full_table = pd.read_csv('data/achat_2014cleaned.csv')
products_table = pd.read_csv('data_cleaned/produits_achats.csv')


table_centrales = full_table[['centraleachat', 'codepanier']].drop_duplicates().groupby(['centraleachat']).count().reset_index()        
centrales = list(table_centrales['centraleachat'].loc[table_centrales['codepanier']> 10000])
centrales.remove('INCONNUE')


tables_households = full_table[['household', 'codepanier']].drop_duplicates().groupby(['household']).count().reset_index()  
households = list(tables_households['household'].loc[tables_households['codepanier']> 10])


table_centrales_final = full_table[['centraleachat', 'household', 'codepanier']].loc[(full_table['centraleachat'].isin(centrales)) &
                                  (full_table['household'].isin(households))].drop_duplicates().groupby(['centraleachat', 'household']).count().reset_index()        

tables_centrale_prep = full_table[['household', 'codepanier']].loc[(full_table['centraleachat'].isin(centrales)) &
                                  (full_table['household'].isin(households))].drop_duplicates().groupby(['household']).count().reset_index()        

tables_centrales_merged = pd.merge(table_centrales_final, tables_centrale_prep, left_on = 'household', right_on = 'household')
tables_centrales_merged['ratio'] = tables_centrales_merged['codepanier_x'] / tables_centrales_merged['codepanier_y']
tables_centrales_merged = tables_centrales_merged.loc[tables_centrales_merged['ratio'] > 0.10]


tables_centrales_count = tables_centrales_merged[['household', 'centraleachat']].groupby(['household']).count().reset_index()
tables_centrales_count_inverted = tables_centrales_count.groupby(['centraleachat']).count().reset_index()

plt.figure(figsize=(10,6), dpi=80)
plt.bar(tables_centrales_count_inverted['centraleachat'], tables_centrales_count_inverted['household'], facecolor='#9999ff', edgecolor='white')
plt.title('Distribution du nombre de centrales achat par foyer')
plt.xlabel('Nombre centrale achat')
plt.ylabel('Nombre de foyers')
plt.savefig('analyse_purchases/hist_centrales_achats.png')
'''

# =============================================================================
# Plot 2 : different distributors as different universes
# =============================================================================

#idee chaque centrale achat est un univers
'''
table_centrale_products_original = full_table[['centraleachat', 'product']].loc[full_table['centraleachat'].isin(centrales)].drop_duplicates()
table_centrale_products = table_centrale_products_original.groupby(['product']).count().reset_index()
table_centrale_products_ones = list(table_centrale_products['product'].loc[table_centrale_products['centraleachat'] == 1])
table_centrale_products_final = table_centrale_products.groupby(['centraleachat']).count().reset_index()

plt.figure(figsize=(10,6), dpi=80)
plt.bar(table_centrale_products_final['centraleachat'], table_centrale_products_final['product'], facecolor='#9999ff', edgecolor='white')
plt.title('Presence du nombre de produits par centrale')
plt.xlabel("Nombre de centrales d'achats distribuant le produit")
plt.ylabel('Nombre de produits')
plt.savefig('analyse_purchases/hist_produits_centrales.png')


table_centrale_unique_products = full_table[['centraleachat', 'product']].loc[(full_table['centraleachat'].isin(centrales)) & (full_table['product'].isin(table_centrale_products_ones))].drop_duplicates()
table_centrale_unique_products = table_centrale_unique_products.groupby(['centraleachat']).count().reset_index()
table_centrale_products_normal = table_centrale_products_original.groupby(['centraleachat']).count().reset_index()
table_centrale_products_normal_merge = pd.merge(table_centrale_unique_products, table_centrale_products_normal, left_on = 'centraleachat', right_on = 'centraleachat')
    
unique_products_centrale =  full_table[['centraleachat', 'product']].loc[(full_table['centraleachat'].isin(centrales)) & (full_table['product'].isin(table_centrale_products_ones))].drop_duplicates()
products_unique_centrale = products_table.loc[products_table['product'].isin(table_centrale_products_ones)]
products_unique_centrale = pd.merge(products_unique_centrale, unique_products_centrale, left_on = 'product', right_on = 'product')

'''


# =============================================================================
# Pdm distributeurs
# =============================================================================
'''
centrale_code = full_table[['centraleachat', 'codeshopachat', 'codepanier']].groupby(['centraleachat', 'codeshopachat']).count().reset_index()

pdm_distributeurs = full_table[['centraleachat', 'codepanier']].loc[full_table['centraleachat'].isin(centrales)].groupby(['centraleachat']).count()

'''

# =============================================================================
# How to aggregate by centraleachat ?
# =============================================================================
'''
# doing some experiments

unique_products_full = pd.merge(products_table, unique_products_centrale, left_on = 'product', right_on = 'product')

jambons = unique_products_full.loc[unique_products_full['sousgroupe'] == 'Jambon Blanc']
glaces = unique_products_full.loc[unique_products_full['sousgroupe'] == 'Glaces']


np.dot(purchase_matrix[:,col_names.index('439')].T, purchase_matrix[:,col_names.index('113967')])[0,0]

v1 = np.ravel(full_table[['qaachat', 'household']].loc[full_table['product'] == 902].groupby(['household']).sum())
v2 = np.ravel(full_table[['qaachat', 'household']].loc[full_table['product'] == 113967].groupby(['household']).sum())
v3 = np.ravel(full_table[['qaachat', 'household']].loc[full_table['product'] == 99239].groupby(['household']).sum())

dict_product_distribution = dict()
for p in prod_list:
    dict_product_distribution[p] = full_table[['qaachat', 'household']].loc[full_table['product'] == p].groupby(['household']).sum()




test = ks_2samp(v1, v2)
test2 = ks_2samp(v1, v3)

prod_list = list(jambons['product'].loc[jambons['count'] > 10])


result = list()
count = 0
for prod1 in prod_list:
    for prod2 in prod_list:
        if prod1 != prod2 :
            v1 = np.ravel(dict_product_distribution[prod1])
            v2 = np.ravel(dict_product_distribution[prod2])

            result.append([str(prod1) +','+ str(prod2), ks_2samp(v1, v2)[1]])
            count +=1
            
            if count % 10000 == 0:
                print(count)
result = pd.DataFrame(result)
ones = result.loc[result[1] > .9]    
pairs = list(ones[0])

all_pairs = dict()
for pr in prod_list:
    for pair in pairs:
        if str(pr) in pair:
    
            for p in pair.split(','):
                if p != str(pr):
                    try:
                        all_pairs[pr].append(p)
                    except:
                        all_pairs[pr] = list()
                        all_pairs[pr].append(p)
                        
    
pairs_331 = sorted(list(set(pairs_331)))
pairs_439 = sorted(list(set(pairs_439)))

pairs_common = list(set(pairs_331).intersection(set(pairs_439)))






counter = 0    
products_table = pd.read_csv('data/produits_achats.csv', encoding='latin1')
dict_qval = create_dict_val_qvol(products_table)
products_table = clean_table(products_table)


products_table = products_table.drop(columns=['valqvol'])
sousgroupes = sorted(list(set(products_table['sousgroupe'])))
table_centrale = create_table_centrale(full_table, products_table, centrales)

dict_to_keep = create_dict_to_keep(sousgroupes)
cols_to_drop = ['fabricant', 'marque'] # 

index = 44
subset = products_table.loc[products_table['sousgroupe'] == sousgroupes[index]]
subset_sousgroupe = transform_table_sousgroupe(subset, sousgroupes[index])
clustering_sousgroupe = clustering_algo_mano(subset, sousgroupes[index], dict_to_keep, dict_qval, full = True)
count_cl = count_clustering_numbers(clustering_sousgroupe, subset, table_centrale)


test = subset.loc[subset['product'].isin(count_cl[1][1])]

'''





'''
len_clusters_dict = dict()
for key in clusters2.keys():
    len_clusters_dict[filter(lambda x: x in printable, key)] = len(clusters2[key])
'''

###############################################################################

#algo clustering 2


'''

def counter_mapping(mapping):
    mapping_final = dict()

    for key in mapping.keys():
        mapping_final[key] = Counter(mapping[key])
        
    return mapping_final

def verify_multiple_belonging(mapping):
    mapping_counter = counter_mapping(mapping)
    list_values = list()
    
    for key1 in mapping_counter.keys():
        for key2 in mapping_counter[key1].keys():
            list_values.append(filter(lambda x: x in printable, key2))
    counter_values = Counter(list_values)
    
    return counter_values


test1 = verify_multiple_belonging(mapping_groupe_produit)
k = test1.keys()[0]
test1[k].keys()


mapping_sousgroupe_produit_final = dict()

for key in mapping_sousgroupe_produit.keys():
    if len(Counter(mapping_sousgroupe_produit[key])) > 1:
        mapping_sousgroupe_produit_final[key] = Counter(mapping_sousgroupe_produit[key])



product_dict = dict()
dim_of_interest = list(products_table.columns.values)
dim_of_interest = [x for x in dim_of_interest if x not in ['product', 'qts','groupe', 'counts', 'nrefs']]

dim_to_round = dim_of_interest[-4:]
for dim in dim_to_round:
    products_table[dim] = np.round(products_table[dim],1)


        
def jaccard_distance(list1, list2):
    intersection = len(list(set(list1).intersection(set(list2))))
    #union = len(list(set(list1).union(set(list2))))
    union = min([len(list1),len(list2)])

    return 1 - (np.float(intersection) / union)


def compute_distance_matrix(product_group_dict, product_list):

    nb_products = len(product_list)
    matrix = np.zeros((nb_products, nb_products))
    for i in range(nb_products):
        for j in range(nb_products):
            if(i<j):
                matrix[i,j] = jaccard_distance(product_group_dict[product_list[i]], product_group_dict[product_list[j]])
    matrix = matrix.T + matrix
    return matrix

def expand_clusters(list_clust_belonging, list_products, nb_clusters):
    clusters = [[] for i in range(nb_clusters)]
    for i in range(len(list_clust_belonging)):
        clusters[list_clust_belonging[i][0]].append(list_products[i])
    return clusters

def sanity_check(cluster_table, group):
    alert_list = list()
    for cluster in cluster_table:
        sous_groupe_list = list()

        for prod in cluster:
            sous_groupe_list.append(filter(lambda x: x in printable, product_dict[group][prod][0]) )
        
        if len(set(sous_groupe_list)) > 1:
            alert_list.append(Counter(sous_groupe_list))
    return alert_list
          
  
def transform_table(table):
    subset_info_nan = len(table) - table.isnull().sum()
    subset_info_zero = table.astype(bool).sum(axis=0)

                 
    headers_of_interest = [header for header in table.columns.values if (subset_info_nan[header] > 0) and (subset_info_zero[header] > 0)]
    subset_final = table[headers_of_interest]
    return subset_final
    


def see_example(list_products):
    
    subset = products_table.loc[products_table['product'].isin(list_products)]
    subset_final = transform_table(subset)
    return subset_final
          
        
    
def get_representative_list(cluster_list):
    representative_list = list()
    for cluster in cluster_list:
        subset = products_table.loc[products_table['product'].isin(cluster)]
        max_count = max(subset['counts'])
        representative = subset.loc[subset['counts'] == max_count]['product'].values[0]
        representative_list.append(representative)
    return representative_list

def compute_dendogram(product_group_dict, product_list):
    matrice_dist = compute_distance_matrix(product_group_dict, product_list)
    
    distArray = ssd.squareform(matrice_dist)
    Z = linkage(distArray, method='ward')
    
    return Z

def display_dimensions_of_interest(group):
    subset = products_table.loc[products_table['groupe'] == group]
    subset_final = transform_table(subset)
    return subset_final
    
def transform_representation_products(group, dimensions_to_throw):
    subset = products_table.loc[products_table['groupe'] == group].reset_index()
    dim_to_keep = list(set(dim_of_interest) - set(dimensions_to_throw))
    dict_group_product = dict()
    for i in range(len(subset)):
        
        final_product = subset[dim_to_keep].loc[i].dropna()
        final_product = final_product[final_product != 0]
        product_list = list()
        for ind in final_product.index:
            interm = ind + ' ' + str(final_product.loc[ind])
            #product_list.append(ind)
            #product_list.append(final_product.loc[ind])
            product_list.append(interm)
        try:
            dict_group_product[subset['product'].loc[i]] = product_list
            
        except:
            dict_group_product[subset['product'].loc[i]] = dict()
            dict_group_product[subset['product'].loc[i]] = product_list
    
    return dict_group_product



###############################################################################

dict_cluster = dict()


#Biscuits
group = 'Biscuits'

overview_group_biscuits = display_dimensions_of_interest(group)
headers_to_throw = [headers_proper[i] for i in [11,23,28,36,43]]
product_dict_biscuits = transform_representation_products('Biscuits', headers_to_throw)
product_biscuits = product_dict_biscuits.keys()

Z = compute_dendogram(product_dict_biscuits, product_biscuits)

fig = plt.figure(figsize=(10, 5))
dn = dendrogram(Z)
plt.show()


nb_clusters = 4

result = cut_tree(Z, n_clusters = nb_clusters)
cluster_expanded = expand_clusters(result, product_biscuits, nb_clusters)
sanity = sanity_check(cluster_expanded, 'Biscuits')


ex_inner = see_example([10242,4101,4105,4106,4107,1978,6233,16474,16493,8389])
ex_inner = see_example([16488,8717,4833,4834,2830,2831,2838,2848,2860,5607,914])
ex_vins = transform_table(products_table.loc[products_table['product'] == 1657])



representative_list = get_representative_list(cluster_expanded)
ex_extended1 = transform_table(products_table.loc[products_table['product'].isin(representative_list)])


'''







'''
#0 analyse de ka distribution des produits et sous produits
groupes = sorted(list(set(products_table['groupe'])))
sousgroupes = sorted(list(set(products_table['sousgroupe'])))


for i in range(len(products_table)):
    if (products_table['sousgroupe'][i] == sousgroupes[168]) and (products_table['groupe'].loc[]== groupes[4]):
        products_table['sousgroupe'][i] = 'Pate_pate'
    elif (products_table['sousgroupe'][i] == sousgroupes[168]) and (products_table['groupe'][i] == groupes[28]):
        products_table['sousgroupe'][i] = 'Pate_charcuterie'

    

groupe_sizes = products_table[['groupe', 'sousgroupe']].groupby('groupe').count()
sousgroup_sizes  = products_table[['groupe', 'sousgroupe']].drop_duplicates().groupby('groupe').count()
count_buy = products_table[['groupe', 'count']].groupby('groupe').sum()

group_info = pd.merge(groupe_sizes, sousgroup_sizes, left_index = True, right_index = True)
group_info = pd.merge(group_info, count_buy, left_index = True, right_index = True)

assert sum(group_info['count']) == sum(products_table['count']) 
assert sum(group_info['sousgroupe_x']) == len(products_table) 
assert sum(group_info['sousgroupe_y']) == len(products_table['sousgroupe'].drop_duplicates())

group_info.to_csv('information_groupe_produits.csv')

sousgroupe_size = products_table[['sousgroupe', 'marque']].groupby('sousgroupe').count()
sousgroupe_buy = products_table[['sousgroupe', 'count']].groupby('sousgroupe').sum()
sousgroupe = pd.merge(sousgroupe_size, sousgroupe_buy, left_index = True, right_index = True)

assert sum(sousgroupe['count']) == sum(products_table['count']) 
assert len(sousgroupe) == len(products_table['sousgroupe'].drop_duplicates())
sousgroupe.to_csv('information_sous_groupe_produits.csv')


#Transormer les tables pour les rendre plus utilisables

##Creation d'un dictionnaire des valeurs renseignées
headers = list(products_table.columns.values)
headers_question = [headers[i] for i in range(11,67,2)]
headers_answer = [headers[i] for i in range(12,68,2)]

mapping_attribute_produit = dict()
mapping_groupe_produit = dict()
mapping_sousgroupe_produit = dict()

printable = set(string.printable)

for index, row in products_table.iterrows():
    headers_present = row[headers_question]
    headers_is_present = list(headers_present.notnull())
    
    headers_present_question = [headers_question[i] for i in range(len(headers_present)) if headers_is_present[i] == True]
    headers_present_answer = [headers_answer[i] for i in range(len(headers_present)) if headers_is_present[i] == True]
    
    headers_question_value = [row[h] for h in headers_present_question]
    headers_answer_value = [row[h] for h in headers_present_answer]
    
    headers_question_value = [filter(lambda x: x in printable, h) for h in headers_question_value]

    
    indexes = np.argsort(headers_question_value)
    
    headers_questions_sorted = [headers_question_value[i] for i in indexes] 
    headers_answer_sorted = [headers_answer_value[i] for i in indexes]
    
    headers_questions_sorted.insert(0, 'product')
    headers_answer_sorted.insert(0,  row['product'])

    headers_question_proper = ';'.join(headers_questions_sorted)
    
    try:
        mapping_attribute_produit[headers_question_proper].append(headers_answer_sorted)
        mapping_groupe_produit[headers_question_proper].append(row['groupe'])
        mapping_sousgroupe_produit[headers_question_proper].append(row['sousgroupe'])

    except:
        mapping_attribute_produit[headers_question_proper] = list()
        mapping_attribute_produit[headers_question_proper].append(headers_answer_sorted)
        
        mapping_groupe_produit[headers_question_proper] = list()
        mapping_groupe_produit[headers_question_proper].append(row['groupe'])
        
        mapping_sousgroupe_produit[headers_question_proper] = list()
        mapping_sousgroupe_produit[headers_question_proper].append(row['sousgroupe'])
'''

    
























'''

subset = products_table.loc[products_table['sousgroupe'] == 'Vins']
subset_final = transform_table(subset)[:10]
subset_final.to_csv('table_vins.csv')
   
agg_product_by_group = dict()
merge_by_group = dict()
counter_list = list()


threshold = 0.8
    
for group in product_dict.keys():
    
    for prod1 in product_dict[group].keys():
        try: 
            agg_product_by_group[group].append(prod1)
            for prod2 in agg_product_by_group[group]:
                    
                if (jaccard_similarity(product_dict[group][prod1], product_dict[group][prod2]) >= threshold) and (prod1 != prod2):
                    try:
                        merge_by_group[group].append([prod1, prod2])
                    except:
                        merge_by_group[group] = list()
                        merge_by_group[group].append([prod1, prod2])
                            
                    break
                     
    
                
        
        except:
            agg_product_by_group[group] = list()
            agg_product_by_group[group].append(prod1)



merge_by_group_clusters = dict()
for group in merge_by_group.keys():
    
    for pair1 in merge_by_group[group]:
        try: 
            merge_by_group_clusters[group].append(pair1)
            for pair2 in merge_by_group_clusters[group]:
                        
                if (len(set(pair1).intersection(set(pair2))) >= 1) and (pair1 != pair2):
    
                    merge_by_group_clusters[group].append(list(set(pair1).union(set(pair2))))
                    merge_by_group_clusters[group].remove(pair1)
                    merge_by_group_clusters[group].remove(pair2)
                    break
                            
    
        
        except:
            merge_by_group_clusters[group] = list()
            merge_by_group_clusters[group].append(pair1)
    
    
products_table_clustered = products_table[:] 
to_drop = list()
for group in merge_by_group_clusters:
    for cluster in merge_by_group_clusters[group]:
        count_total = sum(products_table.loc[products_table['product'].isin(cluster)]['counts'])
        for i, pr in enumerate(cluster):
            if i == 0:
                products_table_clustered.loc[pr]['counts'] = count_total
            else:
                to_drop.append(pr)
    
                    
                
to_keep = list(set(products_table_clustered['product']) - set(to_drop))
products_table_clustered = products_table.loc[products_table['product'].isin(to_keep)]
products_table_pruned = products_table_clustered.loc[products_table_clustered['counts'] > 10]
    





ex1 = see_example([10110,15339,7301,907,9870,911,9872,5009,6546,6547,3908,2204,2205,2206,17834])            
ex2 = see_example([7939,8263,8904,15626,5074,4309,5079,4312,4313,4897,17963,5492,5493,5495,15679])    
ex3 = see_example([16418,4523,3885,4525,7857,5015,18010,5019,5021,5023])
ex4 = see_example([5120,7,8,9,6218,11,16,5137,18,5140,2464,6800,4837,15569,16045,5111,5114,16251,10,5118,9023])

ex_extended1 = transform_table(products_table_pruned.loc[products_table_pruned['sousgroupe'].isin(['Legumes Frais'])])


groupes = products_table['groupe'].drop_duplicates()
sousgroupes = products_table['sousgroupe'].drop_duplicates()

y = np.asarray([[0,0.5,1],[0.5,0,4],[1,4,0]])
distArray = ssd.squareform(y)
Z = linkage(distArray, method='single', metric='euclidean')

fig = plt.figure(figsize=(10, 5))
dn = dendrogram(Z)
plt.show()

result = cut_tree(Z, n_clusters = 2)

group_sizes =group_sizes = products_table[['groupe', 'sousgroupe']].groupby('groupe').count()
sous_group_sizes =group_sizes = products_table[['groupe', 'sousgroupe']].drop_duplicates().groupby('groupe').count()

'''


