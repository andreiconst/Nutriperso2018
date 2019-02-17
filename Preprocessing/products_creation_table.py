#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 15:02:54 2018

@author: andrei
"""

# =============================================================================
# Code file number 0
# Use to create porudct table and clean purchase table (keep only useful columns)
# =============================================================================


import pandas as pd
counter = 0
dict_volumes = dict()

for chunck_df in pd.read_csv('data/achat_2014.csv', chunksize=100000, sep = ';',engine='python',encoding='latin1'):
#    vol = chunck_df[['product', 'valqvol', 'groupe']].groupby(['product', 'valqvol']).count()
#    
#    for index, row in vol.iterrows():
#        try:
#            dict_volumes[index[0]]
#            try:
#                dict_volumes[index[0]][index[1]] += row['groupe']
#            except:
#                dict_volumes[index[0]][index[1]] = row['groupe']
#
#        except:
#            dict_volumes[index[0]] = dict()
#            dict_volumes[index[0]][index[1]] = row['groupe']
      
    
    counter += len (chunck_df)

    if counter == 100000:
        header = list(chunck_df.columns.values[18::])
        header.insert(0, chunck_df.columns.values[13])
        products = chunck_df[header]
        prod_count = products[['product', 'groupe']].groupby('product').count().rename(columns = {'groupe' : 'count'} )
        products = products.drop_duplicates().set_index('product')
        products = pd.concat([products, prod_count], axis = 1)
        purchases_cleaned = chunck_df[list(chunck_df.columns.values[:18])]
    else:
        purchases_cleaned = pd.concat([purchases_cleaned, chunck_df[list(chunck_df.columns.values[:18])]], axis = 0)    
        chunck_df = chunck_df[header]
        chunck_df_count = chunck_df[['product', 'groupe']].groupby('product').count().rename(columns = {'groupe' : 'count'} )
        chunck_df = chunck_df.drop_duplicates().set_index('product')
        chunck_df = pd.concat([chunck_df, chunck_df_count], axis = 1)
    
        
        products = pd.concat([products, chunck_df])

list_products = list()
list_unit_volume = list()
for prod in dict_volumes.keys():
    max_value = 0
     
    for unit_vol in dict_volumes[prod].keys():
        if dict_volumes[prod][unit_vol] > max_value:
            final_unit_vol = unit_vol
            max_value = dict_volumes[prod][unit_vol]

    list_products.append(prod)
    list_unit_volume.append(final_unit_vol)
vol_dataframe = pd.DataFrame(list_unit_volume, index = list_products)          
    




sanity_1 = sum(products['count'])

products_group = pd.DataFrame(products.index).drop_duplicates()
products = pd.merge(products_group, products, left_on = 'product', right_index=True)
products_count = products.groupby('product')['count'].sum()
products = products.drop(['count'], axis = 1)
products = products.drop_duplicates()
products = pd.merge(products, products_count.to_frame(), left_on = 'product', right_index = True)

sanity_2 = sum(products['count'])
assert sanity_1 - sanity_2 == 0
assert sanity_1 - counter == 0
assert len(purchases_cleaned) == counter


heards_test = list(products.columns.values)[2:10]
heards_test.extend(list(products.columns.values)[19:66])

products.to_csv('data/produits_achats.csv')
purchases_cleaned.to_csv('data/achat_2014cleaned.csv')