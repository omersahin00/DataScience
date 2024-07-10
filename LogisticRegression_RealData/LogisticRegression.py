import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("breast_cancer.csv")

x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#Logistic Regression Modelinin Kurulması
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)

#Modelin train set üzerinde eğitilmesi
classifier.fit(x_train, y_train)

#Modelin test edilmesi
y_pred = classifier.predict(x_test)
y_pred_prob = classifier.predict_proba(x_test)

#Confusion Matrix ve AccuracyScore
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

AccuracyScore = accuracy_score(y_test, y_pred)

