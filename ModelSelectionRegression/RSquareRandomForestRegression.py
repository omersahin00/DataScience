import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)


from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(random_state=0, n_estimators=10)

regressor.fit(x_train, y_train)


#Tahmin
y_pred = regressor.predict(x_test)

ComparRandomForestRegression = np.concatenate(
    (
     y_pred.reshape(len(y_pred), 1), 
     y_test.reshape(len(y_test), 1)
    ), 
    1
)


#R Squared Skoru
from sklearn.metrics import r2_score
R2ScoreRandomForest = r2_score(y_test, y_pred)
