"""
import pandas as pd
import matplotlib.pyplot as plt

#Veri okuma
data = pd.read_csv("deneyim-maas.csv", sep=";")
#Grafik oluşturma
plt.scatter(data.deneyim, data.maas)

#Grafik isimlendirme
plt.xlabel("Deneyim")
plt.ylabel("Maaş")
plt.title("Deneyim-Maaş Grafiği")


#Grafik görüntüleme
plt.show()
"""

#Kütüphanelerin import edilmesi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Verilerin import edilmesi
dataset = pd.read_csv("deneyim-maas.csv", sep=";")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


#Train ve Test setlerinin oluşturulması
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)


#Train setleri üzerinden Simple Linear Regression modelinin EĞİTİLMESİ
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)

#Test setileri üzerinden Simple Linear Regression modelinin DENENMESİ
y_predict = lr.predict(x_test)

#Training set sonuçlarının görselleştirilmesi
plt.scatter(x_test, y_test, color = "red") # Veriler üzerinden toploya noktaları yerleştiriyor.
plt.plot(x_train, lr.predict(x_train), color="blue") # X eksenine gerçek verileri yerleştirir, Y eksenine ise tahmin ettiği maaşları yerleştirir ve çizgi çeker.
plt.title("Deneyim-Maaş Grafiği")
plt.xlabel("Deneyim")
plt.ylabel("Maaş")
plt.show()

