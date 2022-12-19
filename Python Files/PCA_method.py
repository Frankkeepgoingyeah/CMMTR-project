# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 10:01:22 2022

@author: zhong
"""

import pandas as pd
import numpy as np
import making_matrix as mm
import datetime
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans





#%% data describe
# full_matrix.describe()
# matrix01.describe()


def obtainfaces(full_matrix,threshold):

    full_matrix=full_matrix.to_numpy()
    # pca method
    var_list = []
    tot_var=0
    number=1
    eigenvalues=[]
    while tot_var<=threshold:
        pca = PCA(n_components=number)
        pca.fit(full_matrix.T)
        var_ratio = pca.explained_variance_ratio_
        tot_var = np.sum(var_ratio)
        var_list.append(tot_var)
        number+=1
    eigenvalues.append(pca.explained_variance_)
    plt.figure(figsize=(8,5))
    plt.plot(list(range(1,number)), var_list)
    plt.grid(linestyle=("--"))
    plt.ylabel("% variance")
    plt.xlabel("No. of components")
    plt.title("Graph of variance against number of components")
    
    #get n component
    components=pca.components_
    
    print('the threshold is %f, we need to keep %d components'%(threshold,components.shape[0]))
    

    # obtain the demand weights of different days
    # average demand and residual demand
    day_average=np.mean(full_matrix,axis=1,keepdims=True)
    day_residual=full_matrix-day_average

    weights=components.dot(day_residual)
    
    return components,weights,day_average,var_list,eigenvalues
    # return components,weights,day_average









#%%#%% clustering and applications

# function returns WSS score for k values from 1 to kmax
def calculate_WSS(points, kmax, clusters):
    
    # elbow method
    Sum_of_squared_distances = []
    K = range(1,kmax)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(points)
        Sum_of_squared_distances.append(km.inertia_)
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Sum of Squared Distances')
    # plt.title('Elbow Method For Optimal k')
    plt.show()
    
    
    
    kmeans = KMeans(n_clusters=clusters, random_state=0).fit(points)
    
    return kmeans


    
   
