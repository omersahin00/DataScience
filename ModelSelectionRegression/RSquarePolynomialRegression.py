import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
pol_reg = PolynomialFeatures(degree=4)

x_pol = pol_reg.fit_transform(x_train)

lr2 = LinearRegression()

lr2.fit(x_pol, y_train)


#Tahmin
y_pred = lr2.predict(pol_reg.transform(x_test))

ComparPolynomialRegression = np.concatenate(
    (
     y_pred.reshape(len(y_pred), 1), 
     y_test.reshape(len(y_test), 1)
    ), 
    1
)


#R Squared Skoru
from sklearn.metrics import r2_score
R2ScorePolynomialRegression = r2_score(y_test, y_pred)
