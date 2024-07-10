import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Bilgisayar_Satis_Tahmin.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

#Logistic Regression Modelinin kurulması
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)

classifier.fit(x_train, y_train)


#Logistic Regression Tahmin denemesi
LogisticRegressionPredict = classifier.predict(ss.transform([[30, 150000]]))
LogisticRegressionPredictKüsuratlı = classifier.predict_proba(ss.transform([[30, 150000]])) #-> Küsüratlı tam değeri gönderir.


#Logistic Regression Test set tahmin denemesi
y_pred = classifier.predict(x_test)


#Confusion Matrix (Hata Matrixi) 
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
AccuracyScore = accuracy_score(y_test, y_pred) # Çıktı: 0.821917808219178

#Train Set sonuçlarının görselleştirilmesi
from matplotlib.colors import ListedColormap

x_set = ss.inverse_transform(x_train)
y_set = y_train #-> Bu değişkeni standart hale getirmemiştik. (sadece 2 adet değişkenden oluştuğu için)

X1, X2 = np.meshgrid(
    np.arange(start = x_set[:, 0].min() - 10, stop = x_set[:, 0].max() + 10, step = 0.25),
    np.arange(start = x_set[:, 1].min() - 1000, stop = x_set[:, 1].max() + 1000, step = 0.25)
)

plt.contourf(X1, X2, classifier.predict(ss.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'blue')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c = ListedColormap(('red', 'blue'))(i), label = j)
plt.title('Logistic Regression - Test Set')
plt.xlabel('Yaş')
plt.ylabel('Maaş')
plt.legend()
plt.show()


# Bu model linear'dir. Yani doğrusal bir öğrenim yapar.
# Verileri bir tabloda gösterdiğimizde ortada çıkan çizgi tahmin ayrım çizgisidir.
# Tahmin verilerini node'ların bu çizginin hangi tarafında bulunduğunda göre karar verir.

