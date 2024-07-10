import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("kalite-fiyat.csv")

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values 

plt.scatter(x, y, color="red")

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x, y)

#Polynomial Regression Model
from sklearn.preprocessing import PolynomialFeatures

degree = 4 #Genelde 4'ü aşılmaz!
pol_reg = PolynomialFeatures(degree = degree) # yazmazsak da derece default 2'dir

x_pol = pol_reg.fit_transform(x)

lr2 = LinearRegression()
lr2.fit(x_pol, y)


#Linear Modelin Görselleştirilmesi
y_linear_predict = lr.predict(x)
plt.scatter(x, y, color = "red")
plt.plot(x, y_linear_predict, color = "blue")
plt.title("Linear")
plt.xlabel("Kalite")
plt.ylabel("Fiyat")
plt.show()


#Polynomial Modelin Görselleştirilmesi
y_polynomial_predict = lr2.predict(x_pol)
plt.scatter(x, y, color = "red")
plt.plot(x, y_polynomial_predict, color = "blue")
plt.title("Polynomial degree={}".format(degree))
plt.xlabel("Kalite")
plt.ylabel("Fiyat")
plt.show()


#Linear Modelin Tahmin Denemesi
linear_predict = lr.predict([[8.5]])


#Polynomial Modelin Tahmin Denemesi
polynomial_predict = lr2.predict(pol_reg.fit_transform([[8.5]]))








