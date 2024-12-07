import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from matplotlib.colors import ListedColormap

#=============================Ön İşleme==================================
dataset = pd.read_csv("Bilgisayar_Satis_Tahmin.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Eksik veri kontrolü ve doldurulması
if dataset.isnull().sum().any():
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    x = imputer.fit_transform(x)
    print("Eksik veriler dolduruldu.")

# Eğitim ve test setine ayırma
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Özellik ölçeklendirme, Normalizasyon
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)
#========================================================================

#==========================Logistic Regression===========================
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)

# Test seti tahminleri
y_pred = classifier.predict(x_test)

# Confusion Matrix ve Accuracy Score
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)
print("\n [[TP FN]\n [FP TN]]")
print("\nAccuracy Score: {:.2f} %".format(accuracy * 100))
#========================================================================


#=============================F1 Score===================================
f1 = f1_score(y_test, y_pred, average='binary')
print("\nF1 Skoru:", f1)
#========================================================================


#========================K-Fold Cross Validation=========================
accuracies = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=10)
print("\nK-Fold Ortalama Doğruluk: {:.2f} %".format(accuracies.mean() * 100))
print("K-Fold Doğruluk Standart Sapması: {:.2f} %".format(accuracies.std() * 100))
#========================================================================


#============================Görselleştirme==============================
x_set = ss.inverse_transform(x_train)
y_set = y_train

X1, X2 = np.meshgrid(
    np.arange(start=x_set[:, 0].min() - 10, stop=x_set[:, 0].max() + 10, step=0.25),
    np.arange(start=x_set[:, 1].min() - 1000, stop=x_set[:, 1].max() + 1000, step=0.25)
)

plt.contourf(X1, X2, classifier.predict(ss.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'blue')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(
        x_set[y_set == j, 0],
        x_set[y_set == j, 1],
        color=ListedColormap(('red', 'blue'))(i),
        label=j
    )

plt.title('Logistic Regression - Eğitim Seti')
plt.xlabel('Yaş')
plt.ylabel('Maaş')
plt.legend()
plt.show()
#========================================================================
