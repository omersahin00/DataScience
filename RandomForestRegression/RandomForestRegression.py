import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("kalite-fiyat.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Random Forest modelinin eğitilmesi
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(random_state = 0, n_estimators = 10)


#Random forest modeli öğrenmesi
regressor.fit(x, y)

#Random forest modeli tahmin denemesi
y_predict_random_forest = regressor.predict([[6.5]])

#Görselleştirme
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))

plt.scatter(x, y, color = "red")
plt.plot(x_grid, regressor.predict(x_grid), color = "blue")
plt.title("Random Forest Modeli")
plt.xlabel("Kalite")
plt.ylabel("Fiyat")
plt.show()
