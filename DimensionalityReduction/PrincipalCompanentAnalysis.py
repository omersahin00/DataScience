import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("breast_cancer.csv")

x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)


#PCA İşlemleri
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)

x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_))

