#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 18:49:39 2018

@author: andrei
"""

#0 importer les librairies et charger les données
import re  
import pandas as pd
import numpy as np
import string

households = pd.read_csv('data/menages_2014.csv', sep=";", encoding='latin1')
printable = set(string.printable)

# =============================================================================
# 0 Travail preliminaire mise en frome des donnes
# =============================================================================


headers = list(households.columns.values)
headers_to_keep = ['household', 'en3', 'en6', 'en15', 'en25', 'fare',  'aiur', 
                   'dpts', 'thab','socc', 'lvai', 'malt', 'cha', 'chie', 'scla', 
                   'rve', 'mor', 'tlpo','tvc1', 'voit','fru1', 'rs1']

households_trimed = households[headers_to_keep]

# =============================================================================
# Creation table 1: donnes pere et mere separees
# Donnes separees entre pere et mere
# =============================================================================


#comprendre index pere et mere de famille

headers_sex = ['ista' + str(i+1) for i in range(15) ]
headers_sex.insert(0, 'household')
subset_ista = households[headers_sex]
subset_ista['index_pere'] = 0
subset_ista['index_mere'] = 0

for i in range(15):
    i+=1
    header_title = str(i) + '_pere'
    subset_ista[header_title] = 0
    subset_ista[header_title].loc[subset_ista['ista' + str(i)] == 2 ] = i
    subset_ista['index_pere'] += subset_ista[header_title]

    header_title = str(i) + '_mere'
    subset_ista[header_title] = 0
    subset_ista[header_title].loc[subset_ista['ista' + str(i)] == 1 ] = i
    subset_ista['index_mere'] += subset_ista[header_title]
    
subset_ista = subset_ista[['household', 'index_mere', 'index_pere']]

household_pm = list()
counter =0

for i in range(len(subset_ista)):
    is_pere = subset_ista['index_pere'][i] > 0 
    is_mere = subset_ista['index_mere'][i] > 0 
    if is_pere == True:
        counter+=1
        etud_pere_header = 'etud' + str(subset_ista['index_pere'][i])
        etud_pere = households[etud_pere_header][i]
        ista_pere_header = 'ista' + str(subset_ista['index_pere'][i])
        ista_pere = households[ista_pere_header][i]
        age_pere_header = 'ageind' + str(subset_ista['index_pere'][i])
        age_pere = households[age_pere_header][i] / 12
        csp_pere_header = 'icsp' + str(subset_ista['index_pere'][i])
        csp_pere = households[csp_pere_header][i] 
        hau_pere_header = 'ihau' + str(subset_ista['index_pere'][i])
        hau_pere = households[hau_pere_header][i] 
        pds_pere_header = 'ipds' + str(subset_ista['index_pere'][i])
        pds_pere = households[pds_pere_header][i]
        itra_pere_header = 'itra' + str(subset_ista['index_pere'][i])
        itra_pere = households[itra_pere_header][i] 
    else:
        etud_pere = np.nan
        ista_pere = np.nan
        age_pere = np.nan
        csp_pere = np.nan
        hau_pere = np.nan
        pds_pere = np.nan
        itra_pere = np.nan
        
    if is_mere == True:
        counter+=1
        etud_mere_header = 'etud' + str(subset_ista['index_mere'][i])
        etud_mere = households[etud_mere_header][i]
        ista_mere_header = 'ista' + str(subset_ista['index_mere'][i])
        ista_mere = households[ista_mere_header][i]
        age_mere_header = 'ageind' + str(subset_ista['index_mere'][i])
        age_mere = households[age_mere_header][i] /12
        csp_mere_header = 'icsp' + str(subset_ista['index_mere'][i])
        csp_mere = households[csp_mere_header][i] 
        hau_mere_header = 'ihau' + str(subset_ista['index_mere'][i])
        hau_mere = households[hau_mere_header][i] 
        pds_mere_header = 'ipds' + str(subset_ista['index_mere'][i])
        pds_mere = households[pds_mere_header][i] 
        itra_mere_header = 'itra' + str(subset_ista['index_mere'][i])
        itra_mere = households[itra_mere_header][i] 
    else:
        etud_mere = np.nan
        ista_mere = np.nan
        age_mere = np.nan
        csp_mere = np.nan
        hau_mere = np.nan
        pds_mere = np.nan
        itra_mere = np.nan

    household_pm.append([subset_ista['household'][i], is_pere, is_mere, etud_pere, ista_pere, age_pere, csp_pere, 
                         itra_pere, hau_pere, pds_pere,
                         etud_mere, ista_mere, age_mere, csp_mere, itra_mere,hau_mere, pds_mere])

household_pm = pd.DataFrame(household_pm, columns = ['household', 'is_pere', 'is_mere', 
                                                     'etud_pere', 'ista_pere', 'age_pere', 'csp_pere',
                                                     'itra_pere', 'hau_pere', 'pds_pere',
                                                     'etud_mere', 'ista_mere', 'age_mere', 'csp_mere',
                                                     'itra_mere','hau_mere', 'pds_mere'])

headers_sex = ['ista' + str(i+1) for i in range(15) ]
def compute_value_columns(datatable, headers, value):
    counter = 0
    for h in headers:
        counter += sum(datatable[h] == value)
    return counter    
    
assert counter == compute_value_columns(households, headers_sex, 1) + compute_value_columns(households, headers_sex, 2)

households_trimed = pd.merge(households_trimed, household_pm, left_on = 'household', right_on = 'household')


# =============================================================================
# 2. Donnees communes pere et mere
# =============================================================================

mapping_dict = dict()


#scla
header = 'scla'
info = list(set(households_trimed[header].values))

try:
    info.remove(np.nan)
except: 
    pass

households_trimed[header] = households_trimed[header].replace(info[1],0)
households_trimed[header] = households_trimed[header].replace(info[2],1)
households_trimed[header] = households_trimed[header].replace(info[4],2)
households_trimed[header] = households_trimed[header].replace(info[0],3)
households_trimed[header] = households_trimed[header].replace(info[3],4)

mapping_dict[header] = [info[1], info[2], info[4], info[0], info[3]]

#chiens et chats et autres nombres
headers_tocategorical = ['chie', 'cha', 'voit', 'fare', 'rs1',
                         'lvai', 'malt', 'mor', 'tlpo', 'tvc1', 'fru1',
                         'en6', 'en15', 'en25']

for header in headers_tocategorical:
    info = list(set(households_trimed[header].values))
    try:
        info.remove(np.nan)
    except: 
        pass
    info = sorted(info)
    
    if info[0][0] == '1':
        info = info[-1:] + info[:-1]
        
    
    for j, token in enumerate(info,0):
        households_trimed[header] = households_trimed[header].replace(token,j)
    mapping_dict[header] = info



#rve
header = 'rve'
info = list(set(households_trimed[header].values))

try:
    info.remove(np.nan)
except: 
    pass
info = sorted(info)

info_interm = [s.replace(" ", "") for s in info] 
info_interm = [re.findall(r'\d+', s) for s in info_interm] 
info_interm = [int(res[0]) for res in info_interm]
indexes = sorted(range(len(info_interm)), key=lambda k: info_interm[k]) 
indexes[0], indexes[1] = indexes[1], indexes[0]

temp_list = list()
for i,j in enumerate(indexes):
    households_trimed[header] = households_trimed[header].replace(info[j],i)
    temp_list.append(info[j])
mapping_dict[header] = temp_list


#en3

header = 'en3'
info = sorted(list(set(households_trimed[header].values)))

try:
    info.remove(np.nan)
except: 
    pass

households_trimed[header] = households_trimed[header].replace(info[0],1)
households_trimed[header] = households_trimed[header].replace(info[1],1)
households_trimed[header] = households_trimed[header].replace(info[2],1)
households_trimed[header] = households_trimed[header].replace(info[3],1)
households_trimed[header] = households_trimed[header].replace(info[4],1)
households_trimed[header] = households_trimed[header].replace(info[5],1)
households_trimed[header] = households_trimed[header].replace(info[6],1)
households_trimed[header] = households_trimed[header].replace(info[7],2)
households_trimed[header] = households_trimed[header].replace(info[8],0)

mapping_dict[header] = [info[8], [info[1],info[2],info[3],info[4],info[5],
                        info[6],info[0]], info[7]]


households_trimed['en6'] = households_trimed['en6'] - households_trimed['en3'] 
households_trimed['en15'] = households_trimed['en15'] - households_trimed['en6'] 
households_trimed['en25'] = households_trimed['en25'] - households_trimed['en15'] 

#socc
header = 'socc'
info = list(set(households_trimed[header].values))

try:
    info.remove(np.nan)
except: 
    pass

households_trimed['proprio'] = 0
households_trimed['proprio'].loc[households_trimed[header] == info[0]] = 1

households_trimed['locataire'] = 0
households_trimed['locataire'].loc[households_trimed[header] == info[2]] = 1

households_trimed = households_trimed.drop(['socc'], axis = 1)



#departement => region
households_trimed['dpts'] = households_trimed['dpts'].str[:2]
#households_trimed.to_csv('data_cleaned/households_trimed.csv', index = False)

# =============================================================================
# Fusion tables peres/mere avec table tout
# =============================================================================
mapped_table = list()

## diviser les données
counter = 0
for i in range(len(households_trimed)):
    if households_trimed['is_pere'][i] == True:
        counter+=1
        age = int(households_trimed['age_pere'][i])
        csp = households_trimed['csp_pere'][i]
        etude = households_trimed['etud_pere'][i]
        itra = households_trimed['itra_pere'][i]
        hau = households_trimed['hau_pere'][i]
        pds = households_trimed['pds_pere'][i]
        sexe = 0
        if households_trimed['is_mere'][i] == True:
            conjoint = 1
        else:
            conjoint = 0
        mapped_table.append([sexe, households_trimed['household'][i], age, csp, etude, itra, hau, pds, conjoint])
    if households_trimed['is_mere'][i] == True:
        counter+=1
        age = int(households_trimed['age_mere'][i])
        csp = households_trimed['csp_mere'][i]
        etude = households_trimed['etud_mere'][i]
        itra = households_trimed['itra_mere'][i]
        sexe = 1
        hau = households_trimed['hau_mere'][i]
        pds = households_trimed['pds_mere'][i]
        if households_trimed['is_pere'][i] == True:
            conjoint = 1
        else:
            conjoint = 0
        mapped_table.append([sexe, households_trimed['household'][i], age, csp, etude, itra, hau, pds, conjoint])


mapping_pm = pd.DataFrame(mapped_table, columns = ['sexe', 'household', 'age', 'csp', 'etude', 'itra', 'hau', 'pds','conjoint'])
mapping_pm['itra'].loc[mapping_pm['itra']==9] = np.nan

hommes_info = mapping_pm.loc[mapping_pm['sexe'] == 0]
femmes_info = mapping_pm.loc[mapping_pm['sexe'] == 1]

assert counter == compute_value_columns(households, headers_sex, 2) + compute_value_columns(households, headers_sex, 1)
assert compute_value_columns(households, headers_sex, 1) == len(femmes_info)
assert compute_value_columns(households, headers_sex, 2) == len(hommes_info)


#fusionner table complete
headers_other = list(households_trimed.columns.values)[0:21]
headers_other.extend(list(households_trimed.columns.values)[37:])
other_information_housholds = households_trimed[headers_other]

hommes_info = pd.merge(hommes_info, other_information_housholds, left_on = 'household', right_on = 'household')
femmes_info = pd.merge(femmes_info, other_information_housholds, left_on = 'household', right_on = 'household')
households_final = pd.concat([hommes_info, femmes_info])

#finaly translate csp into readable format

csp_table = pd.read_csv('data/csp_table.csv')
households_final = pd.merge(households_final, csp_table, left_on = 'csp', right_on = 'csp_code' )
households_final = households_final.drop(['csp', 'csp_code'], axis = 1)
households_final = households_final.rename(columns={'csp_intitule':'csp'})


# =============================================================================
# Sanity checks not messed up
# =============================================================================
assert households_final['household'].drop_duplicates().count() == len(households)
assert compute_value_columns(households, headers_sex, 1) == sum(households_final['sexe'])
assert compute_value_columns(households, headers_sex, 2) == len(households_final) - sum(households_final['sexe'])

# =============================================================================
# Save file
# =============================================================================

households_final.to_csv('data/foyers_traites.csv', index = False)





'''


def graph_plot_variable(data_table, headers, output_folder):
    
    
    
    for header in headers:
    
            
        data_interm = data_table[header].value_counts()
        data_interm = data_interm.append(pd.Series([len(data_table) - sum(data_interm)]))
        labels = list(data_interm.index)
        labels[-1] = 'NA'
        labels = [str(l) for l in labels]
        try:
            labels_proper = [filter(lambda x: x in printable, ind) for ind in labels]
        except:
            labels_proper = labels
    
        index = np.arange(len(labels_proper))
        
        fig = plt.figure()
        plt.xticks(index, labels_proper, fontsize=5, rotation=30)
        plt.title(header)
        plt.bar( index, data_interm.values, align='center', alpha=0.5)
        
        plt.show()
        name = 'graphe_' + header + '.jpeg'
    
        fig.savefig(output_folder + '/' + name, dpi=100) 
        
        

#cycle
header = 'cycle'
info = list(set(households_trimed[header].values))

try:
    info.remove(np.nan)
except: 
    pass

households_trimed['cycle_celib'] = 0
households_trimed['cycle_celib'].loc[households_trimed[header] == info[2]] = 1
households_trimed['cycle_celib'].loc[households_trimed[header] == info[6]] = 2
households_trimed['cycle_celib'].loc[households_trimed[header] == info[8]] = 3

households_trimed['cycle_famille'] = 0
households_trimed['cycle_famille'].loc[households_trimed[header] == info[9]] = 1
households_trimed['cycle_famille'].loc[households_trimed[header] == info[7]] = 2
households_trimed['cycle_famille'].loc[households_trimed[header] == info[5]] = 3
households_trimed['cycle_famille'].loc[households_trimed[header] == info[0]] = 4


households_trimed['cycle_couple'] = 0
households_trimed['cycle_couple'].loc[households_trimed[header] == info[1]] = 1
households_trimed['cycle_couple'].loc[households_trimed[header] == info[4]] = 2
households_trimed['cycle_couple'].loc[households_trimed[header] == info[3]] = 3

households_trimed = households_trimed.drop(['cycle'], axis = 1)


for header in headers_possesseur:
    data[header] = data[header].replace('POSSESSEUR',1)
    data[header] = data[header].replace('NON POSSESSEUR',0)
    mapping_dict[header] = ['NON POSSESSEUR', 'POSSESSEUR']


    
for header in headers_tocategorical:
    info = list(set(data[header].values))
    try:
        info.remove(np.nan)
    except: 
        pass
    info = sorted(info)
    
    if info[0][0] == '1':
        info = info[-1:] + info[:-1]
        
    
    for j, token in enumerate(info,0):
        data[header] = data[header].replace(token,j)
    mapping_dict[header] = info

#net2
header = 'net2'
info = list(set(data[header].values))

try:
    info.remove(np.nan)
except: 
    pass

data[header] = data[header].replace(info[0],0)
data[header] = data[header].replace(info[1],1)
mapping_dict[header] = [info[0], info[1]]


#indo
header = 'indo'
info = list(set(data[header].values))

try:
    info.remove(np.nan)
except: 
    pass

data[header] = data[header].replace(info[2],0)
data[header] = data[header].replace(info[1],1)
data[header] = data[header].replace(info[0],2)
data[header] = data[header].replace(info[3],2)
mapping_dict[header] = [info[2], info[1], info[0], info[3]]

'''
