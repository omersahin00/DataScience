import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Bilgisayar_Satis_Tahmin.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)


#Decision Tree Modelinin Kurulması
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(random_state = 0, criterion = "entropy")


#Modelin eğitilmesi
classifier.fit(x_train, y_train)


#Test seti üzerinde modelin denenmesi
y_pred = classifier.predict(x_test)


#Confusion Matrix ve Accuracy Score
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
AccuracyScoreDT = accuracy_score(y_test, y_pred)


#Test Set Sonuçlarının Görselleştirilmesi
from matplotlib.colors import ListedColormap
x_set = ss.inverse_transform(x_test)
y_set = y_test
X1, X2 = np.meshgrid(np.arange(start =x_set[:, 0].min() -1 , stop=x_set[:, 0].max() + 1, step=0.25),
                     np.arange(start =x_set[:, 1].min() -1 , stop=x_set[:, 1].max() + 1, step=0.25))

plt.contourf(X1, X2, classifier.predict(ss.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'blue')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c = ListedColormap(('red', 'blue'))(i), label = j)
plt.title('Decision Tree (Gini) - Test Set')
plt.xlabel('Yaş')
plt.ylabel('Maaş')
plt.legend()
plt.show()





