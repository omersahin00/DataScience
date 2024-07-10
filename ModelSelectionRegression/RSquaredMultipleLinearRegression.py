import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)


from sklearn.linear_model import LinearRegression

regression = LinearRegression()
regression.fit(x_train, y_train)


#Tahmin
y_pred = regression.predict(x_test)

ComparMultipleLinearRegression = np.concatenate(
    (
     y_pred.reshape(len(y_pred), 1), 
     y_test.reshape(len(y_test), 1)
    ), 
    1
)


#R Squared Skoru
from sklearn.metrics import r2_score
R2ScoreMultipleLinear = r2_score(y_test, y_pred)
