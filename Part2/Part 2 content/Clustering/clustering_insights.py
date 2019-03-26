import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn import cluster
from sklearn import decomposition
from sklearn.metrics import silhouette_samples, silhouette_score

output = pd.read_csv('df_ml_wo_weather.csv' , sep=',')
output=output.loc[output.dep_delay_min>-50]
print(output[:10])
output1=output.loc[output['delay']==1]
output0=output.loc[output['delay']==0]

def ward(data):
    #ward clustering
    #drop na value
    data_clean=data.dropna()
    #print('ward method')

    
    #normalize dataset: prepare for k-means
    #data1=pd.concat([output.clean['wind_speed'], output.clean['visibility'], output.clean['temp'],output.clean['gust_speed'],output.clean['filed_airspeed_kts']], axis=1, keys=['windspeed', 'visibility','temp','gust_speed','airspeed'])
    data1=pd.concat([data_clean['filed_airspeed_kts'],data_clean['dep_delay_min'],data_clean['filed_altitude']], axis=1, keys=['filed_airspeed_kts','filed_altitude','aircraft_manuf'])

    
    x = data1.values
    #minmax = preprocessing.MinMaxScaler()
    #xscale = minmax.fit_transform(x)
    #normalizedDataFrame = pd.DataFrame(x)
    #pprint(normalizedDataFrame[:10])
    
    
    #Use ward to get cluster
    klist = [2,3,4,5,7] #set up a list for k value
    score=[]
    for k in klist: 
        ward = cluster.AgglomerativeClustering(n_clusters=k,linkage='ward')
        cluster_labels = ward.fit_predict(x)
        #Measure Silhouette procedure
        silhouetterate = silhouette_score(x, cluster_labels)
        score.append(silhouetterate)
        print("For ward cluster number k = ", k, "Silhouette_score = :", silhouetterate)
    ind=score.index(max(score))
    ward = cluster.AgglomerativeClustering(n_clusters=klist[ind],linkage='ward')
    cluster_labels = ward.fit_predict(x)
    #plot graph for kmeans cluster for first two class
    plt.scatter(x[:, 0], x[:, 1], c=cluster_labels)
    plt.xlabel('airspeed')
    plt.ylabel('dep_delay_min')
    plt.show()
    
    plt.scatter(x[:, 2], x[:, 1], c=cluster_labels)
    plt.xlabel('altitude')
    plt.ylabel('dep_delay_min')
    plt.show()
    
ward(output)



def kmeans(data):
    #ward clustering
    #drop na value
    data_clean=data.dropna()
    #print('kmeans method')

    
    #normalize dataset: prepare for k-means
    #data1=pd.concat([output.clean['wind_speed'], output.clean['visibility'], output.clean['temp'],output.clean['gust_speed'],output.clean['filed_airspeed_kts']], axis=1, keys=['windspeed', 'visibility','temp','gust_speed','airspeed'])
    data1=pd.concat([data_clean['filed_airspeed_kts'],data_clean['dep_delay_min'],data_clean['filed_altitude']], axis=1, keys=['filed_airspeed_kts','filed_altitude','aircraft_manuf'])

    
    x = data1.values
    #minmax = preprocessing.MinMaxScaler()
    #xscale = minmax.fit_transform(x)
    #normalizedDataFrame = pd.DataFrame(x)
    #pprint(normalizedDataFrame[:10])
    
    
    #Use ward to get cluster
    klist = [2,3,4,5,7] #set up a list for k value
    score=[]
    for k in klist: 
        kmeans = KMeans(n_clusters=k)
        cluster_labels = kmeans.fit_predict(x)
        #Measure Silhouette procedure
        silhouetterate = silhouette_score(x, cluster_labels)
        score.append(silhouetterate)
        print("For kmeans cluster number k = ", k, "Silhouette_score = :", silhouetterate)
    ind=score.index(max(score))
    kmeans = KMeans(n_clusters=klist[ind])
    cluster_labels = kmeans.fit_predict(x)
    #plot graph for kmeans cluster for first two class
    plt.scatter(x[:, 0], x[:, 2], c=cluster_labels)
    plt.xlabel('airspeed')
    plt.ylabel('altitude')
    plt.show()
    
   
    
kmeans(output1)







