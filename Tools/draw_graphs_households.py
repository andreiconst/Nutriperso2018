#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 15:39:19 2018

@author: andrei
"""

#graphes necessaires

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import sys
#sys.path.insert(0, '/home/andrei/Desktop/kantar_2014/code_foyers/')
#from foyers_preprocessing import *

households = pd.read_csv('data/foyers_traites.csv')

#histogramme 

def plot_histogram(datatable, column, title, mapping_label=None, reorder=False):

    data_interm = datatable[column].value_counts()
    
    if mapping_label != None:
        labels_pre = list(data_interm.index)
        labels = list()
        for label in labels_pre:
            labels.append(mapping_label[label])
    else:
        labels = list(data_interm.index)


    if len(datatable) - sum(data_interm) > 0:
        data_interm = data_interm.append(pd.Series([len(datatable) - sum(data_interm)]))
        labels.append('NA')
    
    values = data_interm.values

    if reorder == True:
        indexes = sorted(range(len(labels)), key=lambda k: labels[k])
        labels = [labels[i] for i in indexes]
        values = [values[i] for i in indexes]
        
    labels = [str(l) for l in labels]
    

    
    index = np.arange(len(labels))
    fig = plt.figure(figsize=(10,6), dpi=100)


    plt.xticks(index, labels, fontsize=8, rotation=20)
    plt.title(title)
    plt.bar( index, values, align='center', alpha=0.5)
    plt.bar(index, values, facecolor='#9999ff', edgecolor='white')
    for x,y in zip(index,values):
        plt.text(x, y+0.05, '%.2f' % (float(y) / sum(data_interm)), ha='center', va= 'bottom')
    plt.show()
    name = 'clustering_foyers_docs/' + str(column) + '.jpeg'
    
    fig.savefig(name)
 
'''
households['age_arondi'] = np.round(households['age'] / 5,0) * 5
households['hauteur_arondie'] = np.round(households['hau'] / 5,0) * 5
households['poids_arondi'] = np.round(households['pds'] / 5,0) * 5

    
plot_histogram(households, 'sexe', 'Distribution Sexe personnes de référence',{0:'homme', 1:'femme'})
plot_histogram(households, 'conjoint', 'Distribution statut marital de référence',{0:'sans conjoint', 1:'avec conjoint'})
plot_histogram(households, 'fare', 'Distribution statut familial personnes de référence',{0:'non recomposée', 1:'famille recomposée'})

plot_histogram(households, 'etude', "Distribution niveau d'etude personnes de référence",{0:'Pas encore scolarisé', 
                                                                               1:'Etudes primaires',
                                                                               2:'Etudes secondaires',
                                                                               3:'Technique court',
                                                                               4:'1e,2e, Brevet pro',
                                                                               5:'Technique superieur',
                                                                               6:'Superieur premier cycle',
                                                                               7:'Superieur deuxieme cycle',
                                                                               8:'Superieur troisieme cycle'})
plot_histogram(households, 'scla', 'Distribution Classe sociale personnes de référence',{
        0:'Tres modeste', 1:'Modeste', 2:'Moyenne inferieure', 3:'Moyenne superieure', 4:'Aisee'})
    
plot_histogram(households, 'thab', "Distribution type d'habitation personnes de référence")
plot_histogram(households, 'aiur', "Distribution aires urbaines personnes de référence")
plot_histogram(households, 'age_arondi', "Distribution age personnes de référence", reorder=True)
plot_histogram(households, 'poids_arondi', "Distribution poids personnes de référence", reorder=True)
'''



