import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Bilgisayar_Satis_Tahmin.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

#Logistic Regression Modelinin kurulması
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)

classifier.fit(x_train, y_train)


#Logistic Regression Tahmin denemesi
LogisticRegressionPredict = classifier.predict(ss.transform([[30, 150000]]))
LogisticRegressionPredictKüsuratlı = classifier.predict_proba(ss.transform([[30, 150000]])) #-> Küsüratlı tam değeri gönderir.


#Logistic Regression Test set tahmin denemesi
y_pred = classifier.predict(x_test)


#Confusion Matrix (Hata Matrixi) 
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
AccuracyScore = accuracy_score(y_test, y_pred) # Çıktı: 0.821917808219178

#Train Set sonuçlarının görselleştirilmesi
x_set = ss.inverse_transform(x_train)














