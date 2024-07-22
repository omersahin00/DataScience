import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Mall_Customers.csv")
x = dataset.iloc[:,3:5].values

"""
#Optimum Cluster Sayısının Bulunabilmesi için Diyagram Kullanılması
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x, method = "ward"))
plt.title("Dendogram Ward")
plt.xlabel("Customers")
plt.ylabel("Distances")
"""

#Hierarchical Modelin Kurulması
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, metric = "euclidean", linkage = "ward")

y_hc = hc.fit_predict(x)


#Clusterların Plot Olarak Çizilmesi
plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s = 100, c = "blue", label = "Class 1")
plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s = 100, c = "green", label = "Class 2")
plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s = 100, c = "red", label = "Class 3")
plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s = 100, c = "purple", label = "Class 4")
plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s = 100, c = "yellow", label = "Class 5")
#plt.legend() #-> Grafiğe noktaların ne anlama geldiğini gösteren bir tablo ekler.
plt.title("Müşteri Segmentasyonu - HC")
plt.xlabel("Yıllık Gelir")
plt.ylabel("Harcama Skoru")
plt.show()
