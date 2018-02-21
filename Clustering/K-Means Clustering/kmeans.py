

# K Means Clustering

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the mall dataset with pandas
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

# Using the elbow method to find the optimal number of clusteres
from sklearn.cluster import KMeans
wcss = []

# 1 is included 11 is not
for i in range (1,11) :
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init=10, random_state=0) 
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
    
# Applying K-means to the mall dataset
kmeans = KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0)

# Assign Cluster to each value of X
y_kmeans = kmeans.fit_predict(X )

# Visualising the cluster
plt.scatter(X[y_kmeans==0,0],X[y_kmeans ==0,1],s=100,c='red',label = 'Careful')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans ==1,1],s=100,c='blue',label = 'Standard')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans ==2,1],s=100,c='green',label = 'Target')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans ==3,1],s=100,c='cyan',label = 'Careless')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans ==4,1],s=100,c='magenta',label = 'Sensible')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s = 300,c ='yellow',label='centroid')
plt.title('Clusters of clients')
plt.xlabel('Anual Income (K$)')
plt.ylabel('Spending score (1-100)')
plt.legend()
plt.show()

