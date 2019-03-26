#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 14:50:06 2018

@author: zhoumengzhi
"""

import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn import cluster
from sklearn import decomposition
from sklearn.metrics import silhouette_samples, silhouette_score
#from sklearn.cluster import Ward

#Clustering Analysis 

#read data into python panda.dataframe
output = pd.read_csv('cleandata.csv' , sep=',', encoding='latin1')
#print 1st 10 rows data
print(output[:10])


#ward clustering
def ward():
    #ward clustering
    #drop na value
    output.clean=output.dropna()
    
    print('ward method')

    
    #normalize dataset: prepare for k-means
    #data1=pd.concat([output.clean['wind_speed'], output.clean['visibility'], output.clean['temp'],output.clean['gust_speed'],output.clean['filed_airspeed_kts']], axis=1, keys=['windspeed', 'visibility','temp','gust_speed','airspeed'])
    data1=pd.concat([output.clean['wind_speed'],output.clean['visibility'],output.clean['gust_speed'],output.clean['filed_airspeed_kts']], axis=1, keys=['windspeed','visibility','gust_speed','airspeed'])

    
    x = data1.values
    minmax = preprocessing.MinMaxScaler()
    xscale = minmax.fit_transform(x)
    normalizedDataFrame = pd.DataFrame(xscale)
    pprint(normalizedDataFrame[:10])
    
    
    #Use ward to get cluster
    klist = [2,5,10] #set up a list for k value
    for k in klist: 
        ward = cluster.AgglomerativeClustering(n_clusters=k,linkage='ward')
        cluster_labels = ward.fit_predict(normalizedDataFrame)
    
        #plot graph for kmeans cluster
        pca2D = decomposition.PCA(2)
        plot_columns = pca2D.fit_transform(normalizedDataFrame)
        plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=cluster_labels)
        plt.show()
        
        #Measure Silhouette procedure
        silhouetterate = silhouette_score(normalizedDataFrame, cluster_labels)
        print("For ward cluster number k = ", k, "Silhouette_score = :", silhouetterate)
        
    
ward()
    
#kmeans clustring
def kmeans_silhouetterate():    
    #k-means clustering
    #drop na value
    output.clean=output.dropna()
    
    print('kmeans method')

    
    #normalize dataset: prepare for k-means
    data2=pd.concat([output.clean['wind_speed'],output.clean['visibility'],output.clean['gust_speed'],output.clean['filed_airspeed_kts']], axis=1, keys=['windspeed','visibility','gust_speed','airspeed'])

    x = data2.values
    minmax = preprocessing.MinMaxScaler()
    xscale = minmax.fit_transform(x)
    normalizedDataFrame = pd.DataFrame(xscale)
    pprint(normalizedDataFrame[:10])
    
    #Use k-means to get cluster
    klist = [2,5,10] #set up a list for k value
    for k in klist: 
        kmeans = KMeans(n_clusters=k)
        cluster_labels = kmeans.fit_predict(normalizedDataFrame)
        #centroids = kmeans.cluster_centers_
        #print (centroids)
        
        #plot graph for kmeans cluster
        pca2D = decomposition.PCA(2)
        plot_columns = pca2D.fit_transform(normalizedDataFrame)
        plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=cluster_labels)
        plt.show()
    
        #Measure Silhouette procedure
        silhouetterate = silhouette_score(normalizedDataFrame, cluster_labels)
        print("For k_means cluster number k = ", k, "Silhouette_score = :", silhouetterate)

kmeans_silhouetterate()

#dbscan clustering
def dbscan():
    #dbscan clustering
    #drop na value
    output.clean=output.dropna()
    
    print('dbscan method')
    
    #normalize dataset: prepare for k-means
    data3=pd.concat([output.clean['wind_speed'],output.clean['visibility'],output.clean['gust_speed'],output.clean['filed_airspeed_kts']], axis=1, keys=['windspeed','visibility','gust_speed','airspeed'])

    x = data3.values
    minmax = preprocessing.MinMaxScaler()
    xscale = minmax.fit_transform(x)
    normalizedDataFrame = pd.DataFrame(xscale)
    pprint(normalizedDataFrame[:10])
    
    #Use dbscan to get clster
    elist = [0.05,0.2,0.5] #set up a list for eps value
    for e in elist: 
        dbscan = cluster.DBSCAN(eps=e)
        cluster_labels = dbscan.fit_predict(normalizedDataFrame)
        
        #plot graph for kmeans cluster
        pca2D = decomposition.PCA(2)
        plot_columns = pca2D.fit_transform(normalizedDataFrame)
        plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=cluster_labels)
        plt.show()
            
        #Measure Silhouette procedure
        silhouetterate = silhouette_score(normalizedDataFrame, cluster_labels)
        print("For DBSCAN eps = ", e, "Silhouette_score = :", silhouetterate)

dbscan()      
            
        
        
        
        