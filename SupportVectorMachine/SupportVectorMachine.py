import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Bilgisayar_Satis_Tahmin.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)


#SVM modelinin kurulması
from sklearn.svm import SVC
classifier = SVC()


#Modelin eğitilmesi
classifier.fit(x_train, y_train)


#Tahmin denemesi
SVMTahmin = classifier.predict(ss.transform([[25, 150000]])) #-> Çıktı: 1


#Test set üzerinde sınama
y_pred = classifier.predict(x_test)

