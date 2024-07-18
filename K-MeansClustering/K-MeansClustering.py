import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Mall_Customers.csv")
x = dataset.iloc[:, 3:5].values

#K-Means modelinin kurulmas ve optimum cluster sayısının tespiti
from sklearn.cluster import KMeans

"""
# Cluster sayısı 8 olarak bir deneme yapılıyor:
kmeans = KMeans(n_clusters = 8, init = "k-means++", random_state = 0)   #-> n_clusters default olarak 8'dir
kmeans.fit(x)
print(kmeans.inertia_)
"""


"""
# 1'den 10'a kadar tüm cluster adetleri deneniyor:

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = "k-means++", random_state = 0)   #-> n_clusters default olarak 8'dir
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker = "o")
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()
"""

#Optimum cluster sayısıyla K-Means modelinin kurulması
kmeans = KMeans(n_clusters = 5, init = "k-means++", random_state = 0)
y_kmeans = kmeans.fit_predict(x)
