#importing the required libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.cluster import KMeans

#Loading the iris data
dataset = datasets.load_iris()
dataset = pd.DataFrame(dataset.data,columns = dataset.feature_names)

#Veiwing the Data
dataset.head(10)

#Calculating the "within-cluster sum of square" against clusters range
x = dataset.iloc[:,[0,1,2,3]].values

within_cluster_sum_of_square = []

clusters_range = range(1,11)
for k in clusters_range:
  km = KMeans(n_clusters=k)
  km = km.fit(x)
  within_cluster_sum_of_square.append(km.inertia_)
  
  model = KMeans(n_clusters = 3,init = "k-means++", max_iter= 100, n_init= 10, random_state= 0)
predictions = model.fit_predict(x)

#Visualising the clusters - On the first two columns

plt.scatter(x[predictions == 0,0], x[predictions == 0,1], s = 25, c = 'red', label = 'Iris-setosa')
plt.scatter(x[predictions == 1,0], x[predictions == 1,1], s = 25, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[predictions == 2,0], x[predictions == 2,1], s = 25, c = 'green', label = 'Iris-virginicia')

#Plotting the cluster centers
plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,1], s = 200, c = "black",marker = "*",label = "Centroids")
plt.legend()
plt.show()
