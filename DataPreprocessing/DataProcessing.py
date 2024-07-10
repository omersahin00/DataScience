import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import
dataset = pd.read_csv("Data.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#Kayıp Data
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

#Değişkenlerin düzeltilmesi "OneHotEncoding"
#--> Bu metod birden fazla değeri olan sütunlar için daha uygun.
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [0])], remainder="passthrough")

x = np.array(ct.fit_transform(x))

#Değişkenlerin düzeltilmesi "LabelEncoder"
#--> Bu metod iki adet değeri olan sütunlar için daha uygun. (Yani kategorik değişkenler için daha uygun)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)

#Train ve Test Setleri oluşturma
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

#Feature Scaling (Özellik Ölçekleme)
#Standartizasyon
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

x_train[:, 3:] = ss.fit_transform(x_train[:, 3:])
x_test[:, 3:] = ss.transform(x_test[:, 3:])
