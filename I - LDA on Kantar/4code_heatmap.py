# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import mpl_toolkits
mpl_toolkits.__path__.append('/usr/lib/python3.6/dist-packages/mpl_toolkits/')
import os
os.environ["PROJ_LIB"] = "/home/andrei/anaconda3/share/proj/"; #fixr
from mpl_toolkits.basemap import Basemap

import numpy as np
import math
import matplotlib.pyplot as plt
import shapefile
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
import pandas as pd

import sys
import os
sys.path.insert(0, os.getcwd() + '/Tools/')
from lda_interpretation__bis import *



def lambert932WGPS(lambertE, lambertN):

    class constantes:
        GRS80E = 0.081819191042816
        LONG_0 = 3
        XS = 700000
        YS = 12655612.0499
        n = 0.7256077650532670
        C = 11754255.4261

    delX = lambertE - constantes.XS
    delY = lambertN - constantes.YS
    gamma = math.atan(-delX / delY)
    R = math.sqrt(delX * delX + delY * delY)
    latiso = math.log(constantes.C / R) / constantes.n
    sinPhiit0 = math.tanh(latiso + constantes.GRS80E * math.atanh(constantes.GRS80E * math.sin(1)))
    sinPhiit1 = math.tanh(latiso + constantes.GRS80E * math.atanh(constantes.GRS80E * sinPhiit0))
    sinPhiit2 = math.tanh(latiso + constantes.GRS80E * math.atanh(constantes.GRS80E * sinPhiit1))
    sinPhiit3 = math.tanh(latiso + constantes.GRS80E * math.atanh(constantes.GRS80E * sinPhiit2))
    sinPhiit4 = math.tanh(latiso + constantes.GRS80E * math.atanh(constantes.GRS80E * sinPhiit3))
    sinPhiit5 = math.tanh(latiso + constantes.GRS80E * math.atanh(constantes.GRS80E * sinPhiit4))
    sinPhiit6 = math.tanh(latiso + constantes.GRS80E * math.atanh(constantes.GRS80E * sinPhiit5))

    longRad = math.asin(sinPhiit6)
    latRad = gamma / constantes.n + constantes.LONG_0 / 180 * math.pi

    longitude = latRad / math.pi * 180
    latitude = longRad / math.pi * 180

    return longitude, latitude

def carte_france():

    fig, axes = plt.subplots(1, 1, figsize=(8,8))

    m = Basemap(llcrnrlon=-5, llcrnrlat=40, urcrnrlon=12, urcrnrlat=51,
                resolution='i',projection='cass',lon_0=2.34,lat_0=48,
               ax=axes)
    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents(color='lightgrey', lake_color='#AAAAFF')

    m.drawparallels(np.arange(-40,61.,2.))
    m.drawmeridians(np.arange(-20.,21.,2.))
    m.drawmapboundary(fill_color='#BBBBFF')
    return m, axes


def clean_records(records):
    record_copy = records[:]
    for i in range(len(records)):
        if records[i][1] in['2A','2B']:
            record_copy[i][1] = '20'
    return record_copy
            
# =============================================================================
# 
# =============================================================================


def draw_map(dpts_dataframe,topic_index):
    m, ax = carte_france()
    
    departements = 'data/DEPARTEMENT/DEPARTEMENT.shp'
    shp = departements
    r = shapefile.Reader(shp)
    shapes = r.shapes()
    records = r.records()
    records = clean_records(records)
    done = False
     
    for record, shape in zip(records,shapes):
        geo_points = [lambert932WGPS(x,y) for x, y in shape.points]
        lons = [_[0] for _ in geo_points]
        lats = [_[1] for _ in geo_points]
        data = np.array(m(lons, lats)).T
     
        if len(shape.parts) == 1:
            segs = [data,]
        else:
            # un polygone est une liste de sommets
            # on le transforme en une liste d'arêtes
            segs = []
            for i in range(1,len(shape.parts)):
                index = shape.parts[i-1]
                index2 = shape.parts[i]
                segs.append(data[index:index2])
            segs.append(data[index2:])
     
        lines = LineCollection(segs,antialiaseds=(1,))
        
        # pour changer les couleurs c'est ici, il faudra utiliser le champ records
        # pour les changer en fonction du nom du départements
        if not done:
            for i,_ in enumerate(record):
                print(i,_)
            done = True
    #    dep = retourne_vainqueur(record[2])
        dep = True
        if dep is not None:
            couleur = colors[int(dpts_dataframe['code_couleur'].loc[dpts_dataframe[0] == int(record[1])].values[0]),:]
            lines.set_facecolors(couleur)
            lines.set_edgecolors('k')
            lines.set_linewidth(0.1)
            ax.add_collection(lines)
        else:
            print("--- issue with", record[-1])
    plt.savefig('heatmaps/heatmap_topic' + str(topic_index) +'.png' )

def import_households(list_columns, list_households):
    households = pd.read_csv('data/foyers_traites.csv')
    households = households.loc[households['household'].isin(list_households)]
    households = households[list_columns].drop_duplicates()
    return households

# =============================================================================
# Charger les donnes pertinentes
# =============================================================================


lda_documents_name = 'documents_lda_auto.csv'
lda_topics_name = 'topics_lda_auto.csv'
nb_topics = 30

lda_topics = pd.read_csv('data/' + lda_topics_name, index_col = 0)
lda_documents = pd.read_csv('data/' + lda_documents_name)

products_table = pd.read_csv('data/produits_achats.csv',encoding = "latin1")
products_table = clean_table(products_table)
products_table = products_table.loc[products_table['product'].isin(list(lda_topics.index.values))]

dictionary_description = create_dict_to_keep(sorted(list(products_table['sousgroupe'].drop_duplicates())))

lda_households = np.asarray(lda_documents)[:,1:]
list_households = list(np.asarray(lda_documents)[:,0])
list_households = [int(i) for i in list_households]
households = import_households(['household','dpts','thab'], list_households)

for i in range(30):
    dpts_dataframe = compute_household_variable_mean_lda(i, 'dpts', lda_households, households, list_households)
    dpts_dataframe['code_couleur'] = np.round(dpts_dataframe[1], 1) * 10 + 30
    colors = cm.seismic(np.linspace(0, 1, 60))
    draw_map(dpts_dataframe, i)
    

