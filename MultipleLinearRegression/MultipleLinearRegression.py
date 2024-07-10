import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("50_Startups.csv")

#Bağımlı ve Bağımısz değişkenler ayrılıyor
x = dataset.iloc[:, :-1].values #-> Bağımsız değişkenler
y = dataset.iloc[:, -1].values  #-> Bağımlı değişken

#Değişkenlerin "OneHotEncoding" veya "LabelEncoding" metoduyla düzeltilmesi 
#--> Burada verilerimiz ülkeler olduğundan LabelEncoding kullanacağız.

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [3])], remainder="passthrough")
x = np.array(ct.fit_transform(x))

#Train ve Test setleri
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

#Multiple Linear Regression Modelinin Train set üzerinde öğrenmesi
from sklearn.linear_model import LinearRegression

regression = LinearRegression()
regression.fit(x_train, y_train)

#Modelin test edilmesi
y_predict = regression.predict(x_test)

results_df = pd.DataFrame({"Gerçek Değerler (y_test)": y_test, "Tahmin Edilen Değerler (y_predict)": y_predict, "Aradaki Fark": y_test - y_predict})

