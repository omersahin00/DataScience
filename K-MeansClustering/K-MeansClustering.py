import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator

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

kl = KneeLocator(range(1, 11), wcss, curve="convex", direction="decreasing")
elbow_point = kl.elbow

#Optimum cluster sayısıyla K-Means modelinin kurulması
kmeans = KMeans(n_clusters = elbow_point, init = "k-means++", random_state = 0)
y_kmeans = kmeans.fit_predict(x)
clusters_kordinate = kmeans.cluster_centers_


#Clusterların Plot Olarak Çizilmesi
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = "blue", label = "Class 1")
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = "green", label = "Class 2")
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = "red", label = "Class 3")
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s = 100, c = "purple", label = "Class 4")
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s = 100, c = "yellow", label = "Class 5")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = "black", label = "Centroids")
#plt.legend() #-> Grafiğe noktaların ne anlama geldiğini gösteren bir tablo ekler.
plt.title("Müşteri Segmentasyonu")
plt.xlabel("Yıllık Gelir")
plt.ylabel("Harcama Skoru")
plt.show()

