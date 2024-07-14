import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Bilgisayar_Satis_Tahmin.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)


#Naive Bayes Modelinin kurulması
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()


#Modelin eğitilmesi
classifier.fit(x_train, y_train)


#Tahmin denemesi
NaiveBayesTahmin = classifier.predict(ss.transform([[29, 80000]]))


#Modelin test set üzerinde denenmesi
y_pred = classifier.predict(x_test)


#Confusion Matrix ve Accuracy Score
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
AccuracyScoreNB = accuracy_score(y_test, y_pred) #-> Çıktı: 0.821917808219178


