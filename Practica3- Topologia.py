# -*- coding: utf-8 -*-
"""
Created on Thu May  5 09:01:07 2022

@author: marti
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from sklearn import preprocessing
from sklearn.manifold import TSNE
from math import sqrt

"""
EN TERMINAL LINUX
mkdir practica1
python3 -m venv practica1
source practica1/bin/activate
"""

dataset = pd.read_excel(
    "NBA Stats 202122 All Player Statistics in one Page.xlsx", header=1)
X = dataset[["RPGReboundsRebounds per game.", "SPGStealsSteals per game.",
             "BPGBlocksBlocks per game.", "APGAssistsAssists per game.", "2P%", "3P%", "FT%", "TOPGTurnoversTurnovers per game."]].values
y = dataset["POS"].values


num_objects_train = 100
n_examples_plot = 100
num_objects_test = X.shape[0]-num_objects_train
np.random.seed(1)
random = np.random.choice(X.shape[0], num_objects_train, replace=False)
Index_train = np.isin(range(len(dataset)), random)
X_train = X[Index_train, :]
y_train = y[Index_train]
X_test = X[np.invert(Index_train), :]
y_test = y[np.invert(Index_train)]

X_train_labels = dataset["FULL NAME"][Index_train].values
X_test_labels = dataset["FULL NAME"][np.invert(Index_train)].values

X_embedded = TSNE(n_components=2, init='random').fit_transform(X_train)
colors = {'C': 'red', 'F': 'blue', 'G': 'green',
          'C-F': 'red', 'F-C': 'red', 'G-F': 'green', 'F-G': 'blue'}
cols = [colors[i] for i in y_train]
plt.scatter(X_embedded[:n_examples_plot, 0],
            X_embedded[:n_examples_plot, 1], c=cols[:n_examples_plot], s=100)
for i in range(n_examples_plot):
    txt = X_train_labels[i]
    plt.annotate(txt, (X_embedded[i, 0], X_embedded[i, 1]))
plt.show()

X_clas = np.append(X_train, X_test[:1, :], axis=0)
X_clas = preprocessing.scale(X_clas)
X_clas = TSNE(n_components=2, init='random').fit_transform(X_clas)

# EXERCISE 1
distance_matrix=np.zeros((101,101))
for i in range(101):
    for j in range(101):
        distance_matrix[i,j]+=sqrt((X_clas[i,0]-X_clas[j,0])**2+(X_clas[i,1]-X_clas[j,1])**2)
        
# EXERCISE 2
array_tmp=np.argsort(distance_matrix)
k_nn=array_tmp[:,1:4]

# EXERCISE 3
matriu_adjacencia=np.zeros((len(k_nn),len(k_nn)))
for i in range(len(k_nn)):
    for vei in k_nn[i]:
        if float(i) in k_nn[int(vei)]:
            matriu_adjacencia[i,int(vei)]+=1
# Passar de matriu a graf
# G=nx.from_numpy_matrix(matriu_adjacencies)
