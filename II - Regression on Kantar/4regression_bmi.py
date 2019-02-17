#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 17:00:02 2018

@author: andrei
"""

#regression task using lda topics

import pandas as pd
import numpy as np
import csv
import math

from sklearn import linear_model
from sklearn.svm import LinearSVC
from scipy.sparse import csr_matrix, coo_matrix, csc_matrix
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix

import sys
sys.path.insert(0, '/home/andrei/Desktop/kantar_2014/code_produits/')
sys.path.insert(0, '/home/andrei/Desktop/kantar_2014/code_foyers/')

from tools_regression import *
from foyers_processing import *
from tools_clustering import *
from tools_preprocessing import *

from sklearn.ensemble import RandomForestClassifier
from scipy.sparse.linalg import svds
from sklearn.svm import SVR

from sklearn.linear_model import Lasso
# =============================================================================
# Charger les donnees
'''
1. liste des produits
2. liste des maisonnees
3. table maisonnées
4. table produits

5. matrice raw produits
6. lda
'''


# =============================================================================


households = import_and_transform_households('data_cleaned/foyers_traites.csv')
week_table = import_week_table('data_cleaned/household_activity_week.csv')
circuit_table = import_circuit_table('data_cleaned/household_activity_circuit.csv')

households = merge_households(households, week_table, 
                              circuit_table)


purchase_matrix, row_names_original, col_names_original = load_purchase_matrix('data_cleaned/purchase_table_full.npz', 
                                                                 'data_cleaned/households_matrix_full.txt', 
                                                                 'data_cleaned/products_matrix_full.txt')

products_table = pd.read_csv('data_cleaned/produits_achats.csv', encoding='latin1')
#cluster_table = pd.read_csv('data_cleaned/cluster_products_mano.csv', encoding='latin1')
cluster_table = pd.read_csv('data_cleaned/cluster_products_mano.csv', encoding='latin1') #the other clustering proposed


purchase_matrix_clustered, col_names_clustered = cluster_products(cluster_table, products_table, purchase_matrix, col_names_original)


d = 300
embeddings = pd.read_csv('data_cleaned/embeddings_products_cp_mano.csv')
product_names = list(embeddings['Unnamed: 0'])

index_col = list()
for i, prod in enumerate(product_names):
    index_col.append(col_names_clustered.index(prod))
    
purchase_matrix_clustered = purchase_matrix_clustered[:,index_col]
col_names_clustered = [col_names_clustered[i] for i in index_col]
purchase_matrix_clustered.data = np.log(purchase_matrix_clustered.data + 1)

#prod_weird = purchase_matrix_clustered.max(axis=0).todense()
#
#np.random.seed(123)
#indexes = np.where(prod_weird > 30)[1]
#rand_indexes = np.random.randint(0, len(indexes), 20)
#indexes = indexes[rand_indexes]
#
#products_100 = products_table.loc[products_table['product'].isin([product_names[j] for j in indexes])]
#
#
#for i in indexes:
#    fig = plt.figure()
#    plt.xlim((0,5))
#
#    prod_distrib = np.asarray(purchase_matrix_clustered[:,i].data)
#    plt.hist(prod_distrib, bins= 50)
#
#

#total_purchases = np.ravel(np.sum(purchase_matrix_clustered, axis=1))
#households_relevant = households[[ 'unite_conso', 'nb_weeks']]

#
#dict_info = dict()
#for i, row in households_relevant.iterrows():
#    dict_info[i] = [row['unite_conso'], row['nb_weeks']]
#
#
#weeks_household = list()
#unites_conso = list()
#
#for h in row_names_original:
#    try:
#        weeks_household.append(dict_info[int(h)][1])
#        unites_conso.append(dict_info[int(h)][0])
#    except:
#        weeks_household.append(-1)
#        unites_conso.append(-1)
#        
#relevant_indexes = np.where(np.asarray(weeks_household) >= 0)
#coef_cor_uconso = pearsonr(total_purchases[relevant_indexes], np.asarray(unites_conso)[relevant_indexes])
#coef_cor_weeks = pearsonr(total_purchases[relevant_indexes], np.asarray(weeks_household)[relevant_indexes])

#purchase_matrix_trimed, col_names = trim_sparse_columns(purchase_matrix_clustered, col_names, 30)
purchase_matrix_normalized, row_names_normalized, rej_rows = normalize_rows_purchase_matrix(purchase_matrix_clustered, row_names_original, households,
                                                                                 purge_entropy = True, threshold = 4.5,
                                                                                 binarize_purchases = True)
np.max(purchase_matrix_normalized)
np.mean(purchase_matrix_normalized.data)

#households = households.drop(columns=['nb_weeks', 'MAGASINS POPULAIRES SUPER CHAR'])
#np.max(purchase_matrix_normalized)
# =============================================================================
# Correlation entre bmi d'une meme maisonnee
# =============================================================================

household_1 = households[['bmi']].loc[households['sexe']==1]
household_0 = households[['bmi']].loc[households['sexe']==0]
    
correlation_bmi = pd.merge(household_1, household_0, left_index = True, right_index = True).dropna()
bmi_1 = list(correlation_bmi['bmi_x'])
bmi_0 = list(correlation_bmi['bmi_y'])
coef_cor = pearsonr(bmi_0, bmi_1)


#load embeddings



kmeans = KMeans(n_clusters = 1500, random_state = 123)
clusters = kmeans.fit(embeddings)
clusters_embedding = cluster_representation(clusters.labels_, product_names, products_table)

purchase_matrix_kmeans, col_names_kmeans = cluster_products(clusters_embedding, products_table, purchase_matrix_normalized, col_names_clustered)

purchase_matrix_final, row_names_final, rej_rows = normalize_rows_purchase_matrix(purchase_matrix_clustered, row_names_original, households,
                                                                                 purge_entropy = True, threshold = 4.5,
                                                                                 binarize_purchases = True)

purchase_matrix_log = log_sparse(purchase_matrix_clustered)
np.max(purchase_matrix_clustered)

#purchase_matrix_final, col_final = delete_max_col(purchase_matrix_clustered, col_names_clustered, 10)




purchase_matrix_final, row_names_final= delete_rows_entropy(purchase_matrix_clustered, row_names_original, 4)
#purchase_matrix_final, row_names_final= purchase_matrix_normalized, row_names_normalized
purchase_matrix_total = np.sum(purchase_matrix_final, axis= 1)
purchase_matrix_final = purchase_matrix_final.multiply(1/purchase_matrix_total)
purchase_matrix_max = purchase_matrix_final.max(axis=1).todense()
purchase_matrix_final = purchase_matrix_final.multiply(1/purchase_matrix_max)



purchase_matrix_final = purchase_matrix_final.tocsr()


features_matrix, regression_target, regression_id, sample_weights = create_features_table(purchase_matrix_final, row_names_final, 
                                                                               households, correlation_bmi, 
                                                                               mode= 'raw', 
                                                                               discretize_target = False, include_socio = False,
                                                                               predict_worse = True)


features_matrix, regression_target, regression_id = create_features_table_da(purchase_matrix_final, row_names_final, households, 'gender')


# =============================================================================
# Prediction errors
# =============================================================================

a_1 = [3e2, 1e3, 3e3, 1e4]
a_2 = [10, 30, 1e2, 3e2]
a_3 = [3e-6,1e-5]
a_4 = [1e-2, 3e-2]

for a in a_3:
#    regr = linear_model.Ridge(alpha=a)
    regr = Lasso(alpha=a)
    scores_train, scores_test = custom_cross_validation(regr, features_matrix, regression_target)
    print(np.mean([scores_train[i][0] for i in range(len(scores_train))]))
    print(np.mean([scores_train[i][1] for i in range(len(scores_train))]))
    
    print(np.mean([scores_test[i][0] for i in range(len(scores_test))]))
    print(np.mean([scores_test[i][1] for i in range(len(scores_test))]))

    print('------------------------------------------------------------')


model = linear_model.Lasso(alpha=3e-5)
model.fit(features_matrix, regression_target)
coef_normal = model.coef_[:len(product_names)]
coef_table = pd.DataFrame([coef_normal], columns = product_names).T
coef_table.to_csv('coeficients_multivaries.csv')

coef1 = pd.read_csv('coeficients_multivaries.csv', index_col=0)
coef2 = pd.read_csv('coeficients_monovaries.csv', index_col=0)
coef_final = pd.merge(coef1, coef2, left_index = True, right_index = True)
coef_final = pd.merge(products_table, coef_final, left_on = 'product', right_index = True)
coef_final.to_csv('coef_final.csv')

coef_manipulate = pd.read_csv('coef_merged.csv', index_col=0)
coef_manipulate = coef_manipulate.loc[(coef_manipulate['rank_x'] <=200) | ((coef_manipulate['rank_y'] <=200)) | 
        (coef_manipulate['rank_x'] >= + np.max(coef_manipulate['rank_x']) - 200) | (coef_manipulate['rank_y'] >= + np.max(coef_manipulate['rank_y']) - 200)]
coef_manipulate = coef_manipulate.to_csv('coef_manipulate.csv')
coef_manipulate = pd.read_csv('coef_manipulate.csv', index_col=0)


plt.figure(figsize=(10,6), dpi=100)    
plt.scatter(coef_manipulate['rank_x'], coef_manipulate['rank_y'],marker='.', s = 0.5, c='black', alpha=0.5)
plt.title('Multivariate vs univariate coeficients')
plt.savefig('multi_uni_coef')




coef_male = model.coef_[len(product_names):len(product_names)*2]
coef_female = model.coef_[len(product_names)*2:len(product_names)*3]
coef_all = pd.DataFrame([coef_normal, coef_male, coef_female], columns = product_names).T
coef_all['difference'] = coef_all[1] - coef_all[2]
products_trimmed = products_table.loc[products_table['product'].isin(product_names)]
products_coef = pd.merge(products_trimmed, coef_all, left_on = 'product', right_index = True)
products_coef.to_csv('products_coefficients.csv')

#Plot comparing BMI in a couple (men vs women)
households_pruned = households.dropna(subset=['bmi','conjoint', 'sexe'])
bmi_male_couple = households_pruned[['bmi']].loc[(households_pruned['conjoint']==1)&(households_pruned['sexe']==0)]
bmi_female_couple = households_pruned[['bmi']].loc[(households_pruned['conjoint']==1)&(households_pruned['sexe']==1)]
bmi_full = pd.merge(bmi_male_couple, bmi_female_couple, left_index = True, right_index = True)


    plt.figure(figsize=(10,6), dpi=100)    
plt.scatter(bmi_full['bmi_x'], bmi_full['bmi_y'],marker='.', s = 0.5, c='black', alpha=0.5)
plt.title('Men vs women BMI in couples')
plt.save('men_women_BMI')

####

groupes = sorted(set(list(products_table['groupe'])))
groupes = [groupes[i] for i in [17,6,0,15,14,8,5,34,22,27,19,23,29,10,25,26,12,31,20,1]]

colors = cm.tab20(np.linspace(0, 1, len(groupes)))

products_coef_trimed = products_coef.loc[(np.abs(products_coef[1])>0) | (np.abs(products_coef[2])>0)]
plt.figure(figsize=(20,12), dpi=200)    

fig, ax = plt.subplots()
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling


for i, l in enumerate(groupes):

    subset = products_coef_trimed.loc[products_coef_trimed['groupe'] == l]
    ax.scatter(subset[1], subset[2], marker='.', c=colors[i,:], alpha=0.5)


#ax.legend(loc='upper right', prop={'size': 7})
path = 'coef_male_vs_female_nolegend.png'
plt.savefig(path, dpi=200)


#Plot comparing coeficients colored by groupe
#


coef_highest = np.argsort(-coef)[:20]
coef_lowest = np.argsort(coef)[:20]
coef
products_highest = products_table.loc[products_table['product'].isin([product_names[j] for j in coef_highest])]
products_lowest = products_table.loc[products_table['product'].isin([product_names[j] for j in coef_lowest])]

products_highest.to_csv('high_coef_lasso.csv')
products_lowest.to_csv('low_coef_lasso.csv')

table_coef = pd.DataFrame(coef, index = col_names_clustered)
table_coef.to_csv('data_cleaned/coef_regression.csv', index = True)




coef_matrix = vizalisation_coefficients(col_names, coef, products_table, 10)
coef_matrix.to_csv('regression_results/coeficient_matrix25k.csv')

model = linear_model.Ridge(alpha=1000)
model.fit(features_matrix, regression_target)
coef = model.coef_
description_produits = get_description_products(products_clusters, dict_description, products_table)
description_produits.extend(households_label)


def save_coefs(coef, description):
    with open('regression_results/regression_coefficients.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',' )

        for i in range(len(coef)):
            writer.writerow([description[i], coef[i]])
save_coefs(coef, description_produits)


prediction = model.predict(features_matrix)
error = np.asarray(regression_target) - prediction
fig = plt.figure(figsize=(10,6), dpi=100)
np.random.seed(123)
random_index = list(np.random.randint(0, len(error), 1000))
plt.scatter([regression_target[i] for i in random_index], [error[i] for i in random_index], s=10)
fig.savefig('regression_results/error_regression_raw_.png')



entropies = np.round(rej, 1)
from collections import Counter
entropy_freq = Counter(entropies)

ordered_keys = sorted(list(entropy_freq.keys()))
array_frequ = np.zeros(len(ordered_keys))
for i,key in enumerate(ordered_keys):
    array_frequ[i] = entropy_freq[key]
total_hh = np.sum(array_frequ)
frequ_entropy = np.cumsum(array_frequ)
frequ_entropy /= total_hh
#
# =============================================================================
# Entrainer les modeles 
# =============================================================================
#normalized : 1e-1 0.09
#binaried :    1000  
#non norm : 10000               


for k in [10,20,30,50,100,200,300,500,1000]:
    for j in [3,4,5,6]:
        clf = RandomForestRegressor(n_estimators = 100, random_state=123, n_jobs=-1, max_depth=j, max_features=k)
        scores_train, scores_test = custom_cross_validation(clf, features_matrix, regression_target)
        print(np.mean([scores_train[i][0] for i in range(len(scores_train))]))
        print(np.mean([scores_train[i][1] for i in range(len(scores_train))]))
        
        print(np.mean([scores_test[i][0] for i in range(len(scores_test))]))
        print(np.mean([scores_test[i][1] for i in range(len(scores_test))]))
    
        print('------------------------------------------------------------')

# =============================================================================
# 
# =============================================================================



iterator = 0

for do_worse in [True, False]:
    for purge_threshold in [0,2.5, 3.0,3.5,4,4.3,4.6,5]:
        for features in [['socio-eco', '', ''], ['raw', True,'normalize'],['raw', True,'binarize'],['raw', False,'normalize'], ['raw', False,'binarize']]:
                    
                if (features[0] == 'raw') and(features[2] == 'binarize'):
                    binarize = True
                else:
                    binarize = False
                    
                features_matrix, regression_target, regression_id, rej = create_features_table(mode=features[0], binarize_matrix = binarize,
                                                                                               discretize_target =False, predict_worse=do_worse,
                                                                                                       purge_missing = True, include_socio = features[1],
                                                                                                       threshold = purge_threshold)     
                    
                    
                for a in [300,1000,3000,10000]:
                        
                    regr = linear_model.Ridge(alpha=a)
                    scores_train, scores_test = custom_cross_validation(regr, features_matrix, regression_target, classification=False)
                    
                    r2_train = np.mean([scores_train[i][0] for i in range(len(scores_train))])
                    sq_error_train = np.mean([scores_train[i][1] for i in range(len(scores_train))])

                        
                    r2_test = np.mean([scores_test[i][0] for i in range(len(scores_test))])
                    sq_error_test = np.mean([scores_test[i][1] for i in range(len(scores_test))])
                        
                    size_data = len(regression_target)
                        
                    if iterator == 0:
                        with open('regression_results/results_regression.csv', 'w') as csvfile:
                            writer = csv.writer(csvfile, delimiter=',' )
                            writer.writerow(['features', 'add socio', 'preprocessing', 'predict_worse', 
                                             'extreme_only', 'threshold','alpha', 'recall_train',
                                              'precision_train', 'f1_train','recall_test', 
                                             'precision_test', 'f1_test', 'precision_baseline', 'f1_baseline'])        
                            writer.writerow([features[0], str(features[1]), str(features[2]), 
                                             str(do_worse), str(purge_threshold), str(a),
                                             r2_train, sq_error_train, r2_test, 
                                             sq_error_test,  size_data])         
                    else:
                        with open('regression_results/results_regression.csv', 'a') as csvfile:
                            writer = csv.writer(csvfile, delimiter=',' )
                            writer.writerow([features[0], str(features[1]), str(features[2]), 
                                             str(do_worse), str(purge_threshold), str(a),
                                             r2_train, sq_error_train, r2_test, 
                                             sq_error_test,  size_data]) 
                    iterator +=1









# =============================================================================
# 
# =============================================================================

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

np.random.seed(123)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
# =============================================================================
# 
# =============================================================================



clf = RandomForestRegressor(n_estimators = 500, random_state=123, n_jobs=-1, max_depth=16, max_features=8)
clf_final = clf.fit(features_matrix, regression_target)
index = np.argsort(-clf.feature_importances_)[:5]

features_rf = list()
for i in index:
    features_rf.append([i, clf.feature_importances_[i]])

#linear svc
    #1e-1 for linear svc
#normalize [1e-3,3e-3,1e-2,3e-2,1e-1, 3e-1,1,3]
#binarize 
for k in [1e-3,3e-3,1e-2,3e-2,1e-1, 3e-1,1,3]:
    regr = linear_model.LogisticRegression(C=k, random_state=123)
    scores_train, scores_test = custom_cross_validation(regr, features_matrix, regression_target, classification=True)
    print(np.mean([scores_train[i][0] for i in range(len(scores_train))]))
    print(np.mean([scores_train[i][1] for i in range(len(scores_train))]))
    print(np.mean([scores_train[i][2] for i in range(len(scores_train))]))
    
    print(np.mean([scores_test[i][0] for i in range(len(scores_test))]))
    print(np.mean([scores_test[i][1] for i in range(len(scores_test))]))
    print(np.mean([scores_train[i][2] for i in range(len(scores_train))]))

    print('------------------------------------------------------------')

random = sum(regression_target) / len(regression_target)

# =============================================================================
# 
# =============================================================================

X2 = sm.add_constant(features_matrix)
est = sm.OLS(regression_target, X2)
est2 = est.fit()
print(est2.summary())
summary = str(est2.summary())

with open("regression_results/regression_lda30_nodemo.txt", "w") as f:
    f.write(summary)


# =============================================================================
# Error analysis    
# =============================================================================
    
regr = linear_model.LinearRegression()
regr.fit(features_matrix, regression_target)
y_pred = regr.predict(features_matrix)

np.max(y_pred) 
    
error = regression_target - y_pred
error_sq = np.abs(error)
np.mean(error_sq)

purchases_quantity = list()
for i in range(len(regression_id)):
    purchases_quantity.append(dict_household_purchases[regression_id[i][:-1]])

result = pearsonr(error_sq, purchases_quantity)
result2 = spearmanr(error_sq, purchases_quantity)

fig = plt.figure(figsize=(10,6), dpi=100)
np.random.seed(123)
random_index = list(np.random.randint(0, len(error_sq), 1000))
plt.scatter([error_sq[i] for i in random_index], [purchases_quantity[i] for i in random_index], s=10)
fig.savefig('regression_results/error_regression.png')


list_households_int = [int(h) for h in list_households]
list_households_datatable = list(set(households['household'].dropna()))

diff = list(set(list_households_datatable) - set(list_households_int))
rej_households = list(set([int(row['household']) for row in rej]))

why_rej = list(set(rej_households) - set(diff))
row = households.loc[households['household'] == 47104]



subset_old = households[['age', 'bmi']].loc[households['age'] > 60].reset_index()
subset_young = households[['age', 'bmi']].loc[households['age'] < 30].reset_index()

subset_male = households[['age', 'bmi']].loc[households['sexe']  == 0].reset_index()
subset_female = households[['age', 'bmi']].loc[households['sexe'] == 1].reset_index()

subset_rich = households[['age', 'bmi']].loc[(households['scla'])  == 4 & (households['age'] > 30) & (households['age'] < 50)].reset_index()
subset_poor = households[['age', 'bmi']].loc[(households['scla'] == 1) & (households['age'] > 30) & (households['age'] < 50)].reset_index()

subset_educated = households[['age', 'bmi']].loc[(households['etude'].isin([7,8])) & (households['age'] > 30) & (households['age'] < 50)].reset_index()
subset_undeducated = households[['age', 'bmi']].loc[(households['etude'].isin([3,6])) & (households['age'] > 30) & (households['age'] < 50)].reset_index()

def compare_cumulative_curves(subset1, subset2, title1, title2, title_all, save_name):
    np.random.seed(123)
    indexes_random1 = np.random.randint(0, len(subset1), 1000)
    indexes_random2 = np.random.randint(0, len(subset2), 1000)
    
    array1 = np.asarray(subset1['bmi'].loc[indexes_random1])
    array2 = np.asarray(subset2['bmi'].loc[indexes_random2])

    values1, base = np.histogram(array1, bins=100)
    values2, base = np.histogram(array2, bins=100)
    
    #evaluate the cumulative
    cumulative1 = np.cumsum(values1)
    cumulative2 = np.cumsum(values2)
    
    # plot the cumulative function
    plt.plot(base[:-1], cumulative1/10, c='blue', label= title1)
    #plot the survival function
    plt.plot(base[:-1],cumulative2/10 , c='green', label = title2)
    plt.axvline(x=18.5)
    plt.axvline(x=25)
    plt.axvline(x=30)
    plt.title(title_all)
    plt.legend()
    plt.savefig('clustering_foyers_docs/' + save_name )

compare_cumulative_curves(subset_educated, subset_undeducated, 'educated', 'uneducated', 'Cumulative distribution comparison: educate vs. uneducated', 
                          'etud_non_cumul.png')

compare_cumulative_curves(subset_old, subset_young, 'old', 'young', 'Cumulative distribution comparison: old vs. young', 
                          'y_o_cumul.png')

compare_cumulative_curves(subset_male, subset_female, 'male', 'female', 'Cumulative distribution comparison: male vs. female', 
                          'm_f_cumul.png')

compare_cumulative_curves(subset_rich, subset_poor, 'rich', 'poor', 'Cumulative distribution comparison: rich vs. poor', 
                          'r_p_cumul.png')

 =============================================================================
 Graph bmi
 =============================================================================
values = np.round(np.asarray(households['bmi']),0)
    
plt.figure(figsize=(10,6), dpi=80)
plt.title('Distribution of BMI in population')
plt.xlabel('BMI value')
plt.ylabel('% of population')
plt.hist(values, bins =48, density = True)
plt.savefig('clustering_foyers_docs/BMI_distribution.png')
