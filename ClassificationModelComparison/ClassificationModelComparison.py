import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from matplotlib.colors import ListedColormap

#Sınıflandırma Algoritmaları
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

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
lr_classifier = LogisticRegression(random_state=0)
lr_classifier.fit(x_train, y_train)

# Test seti tahminleri
y_pred_lr = lr_classifier.predict(x_test)

# Confusion Matrix ve Accuracy Score
cm_lr = confusion_matrix(y_test, y_pred_lr)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print("\nAccuracy Score (Logistic Regression): {:.2f} %".format(accuracy_lr * 100))
#========================================================================


#========================Decison Tree Classifier=========================
dt_classifier = DecisionTreeClassifier(random_state=0, criterion='entropy')
dt_classifier.fit(x_train, y_train)

# Test seti tahminleri
y_pred_dt = dt_classifier.predict(x_test)

# Confusion Matrix ve Accuracy Score
cm_dt = confusion_matrix(y_test, y_pred_dt)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("\nAccuracy Score (Decision Tree): {:.2f} %".format(accuracy_dt * 100))
#========================================================================


#========================K-Neighbors Classifier=========================
kn_classifier = KNeighborsClassifier(n_neighbors = 5)
kn_classifier.fit(x_train, y_train)

# Test seti tahminleri
y_pred_kn = kn_classifier.predict(x_test)

cm_kn = confusion_matrix(y_test, y_pred_kn)
accuracy_kn = accuracy_score(y_test, y_pred_kn)
print("\nAccuracy Score (K-Neighbors): {:.2f} %".format(accuracy_kn * 100))
#========================================================================


#==============================Naive Bayes===============================
nb_classifier = GaussianNB()
nb_classifier.fit(x_train, y_train)

# Test seti tahminleri
y_pred_nb = nb_classifier.predict(x_test)

cm_nb = confusion_matrix(y_test, y_pred_nb)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print("\nAccuracy Score (Naive Bayes): {:.2f} %".format(accuracy_nb * 100))
#========================================================================


#========================Support Vector Machine==========================
svm_classifier = SVC(probability=True, kernel="rbf")
svm_classifier.fit(x_train, y_train)

# Test seti tahminleri
y_pred_svm = svm_classifier.predict(x_test)

cm_svm = confusion_matrix(y_test, y_pred_svm)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("\nAccuracy Score (Support Vector Machine): {:.2f} %".format(accuracy_svm * 100))
#========================================================================


#=============================F1 Score===================================
# Logistic Regression
f1_lr = f1_score(y_test, y_pred_lr, average="binary")
print("\n\nF1 Skoru (Logistic Regression):", f1_lr)

# Decision Tree
f1_dt = f1_score(y_test, y_pred_dt, average="binary")
print("\nF1 Skoru (Decision Tree):", f1_dt)

# K-Neighbors
f1_kn = f1_score(y_test, y_pred_kn, average="binary")
print("\nF1 Skoru (K-Neighbors):", f1_kn)

# Naive Bayes
f1_nb = f1_score(y_test, y_pred_nb, average="binary")
print("\nF1 Skoru (Naive Bayes):", f1_nb)

# Support Vector Machine
f1_svm = f1_score(y_test, y_pred_svm, average="binary")
print("\nF1 Skoru (Support Vector Machine)", f1_svm)
#========================================================================


#========================K-Fold Cross Validation=========================
# Logistic Regression
accuracies_lr = cross_val_score(estimator = lr_classifier, X = x_train, y = y_train, cv=10)
print("\n\nK-Fold Ortalama Doğruluk (Logistic Regression): {:.2f} %".format(accuracies_lr.mean() * 100))
print("K-Fold Doğruluk Standart Sapması (Logistic Regression): {:.2f} %".format(accuracies_lr.std() * 100))

# Decision Tree
accuracies_dt = cross_val_score(estimator=dt_classifier, X=x_train, y=y_train, cv=10)
print("\nK-Fold Ortalama Doğruluk (Decision Tree): {:.2f} %".format(accuracies_dt.mean() * 100))
print("K-Fold Doğruluk Standart Sapması (Decision Tree): {:.2f} %".format(accuracies_dt.std() * 100))

# K-Neighbors
accuracies_kn = cross_val_score(estimator=kn_classifier, X=x_train, y=y_train, cv=10)
print("\nK-Fold Ortalama Doğruluk (K-Neighbors): {:.2f} %".format(accuracies_kn.mean() * 100))
print("K-Fold Doğruluk Standart Sapması (K-Neighbors): {:.2f} %".format(accuracies_kn.std() * 100))

# Naive Bayes
accuracies_nb = cross_val_score(estimator=nb_classifier, X=x_train, y=y_train, cv=10)
print("\nK-Fold Ortalama Doğruluk (Naive Bayes): {:.2f} %".format(accuracies_nb.mean() * 100))
print("K-Fold Doğruluk Standart Sapması (Naive Bayes): {:.2f} %".format(accuracies_nb.std() * 100))

# Support Vector Machine
accuracies_svm = cross_val_score(estimator=svm_classifier, X=x_train, y=y_train, cv=10)
print("\nK-Fold Ortalama Doğruluk (Support Vector Machine): {:.2f} %".format(accuracies_svm.mean() * 100))
print("K-Fold Doğruluk Standart Sapması (Support Vector Machine): {:.2f} %".format(accuracies_svm.std() * 100))
#========================================================================


#=========================En İyi Model Tespiti===========================
scores = {
    "Logistic Regression": {
        "accuracy": accuracy_lr,
        "f1_score": f1_lr,
        "k_fold_mean": accuracies_lr.mean() * 100,
        "k_fold_std": accuracies_lr.std() * 100
    },
    "Decision Tree": {
        "accuracy": accuracy_dt,
        "f1_score": f1_dt,
        "k_fold_mean": accuracies_dt.mean() * 100,
        "k_fold_std": accuracies_dt.std() * 100
    },
    "K-Neighbors": {
        "accuracy": accuracy_kn,
        "f1_score": f1_kn,
        "k_fold_mean": accuracies_kn.mean() * 100,
        "k_fold_std": accuracies_kn.std() * 100
    },
    "Naive Bayes": {
        "accuracy": accuracy_nb,
        "f1_score": f1_nb,
        "k_fold_mean": accuracies_nb.mean() * 100,
        "k_fold_std": accuracies_nb.std() * 100
    },
    "Support Vector Machine": {
        "accuracy": accuracy_svm,
        "f1_score": f1_svm,
        "k_fold_mean": accuracies_svm.mean() * 100,
        "k_fold_std": accuracies_svm.std() * 100
    }
}

weights = {
    "accuracy": 0.5,
    "f1_score": 0.3,
    "kfold_mean": 0.2
}

def normalize(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)

# Modellerin Skorlarını hesaplama
final_scores = {}
for model, metrics in scores.items():
    # Dinamik olarak her metrik için minimum ve maksimumu belirleme
    accuracy_values = [metrics["accuracy"] for metrics in scores.values()]
    f1_values = [metrics["f1_score"] for metrics in scores.values()]
    kfold_values = [metrics["k_fold_mean"] for metrics in scores.values()]
    
    accuracy_min, accuracy_max = min(accuracy_values), max(accuracy_values)
    f1_min, f1_max = min(f1_values), max(f1_values)
    kfold_min, kfold_max = min(kfold_values), max(kfold_values)
    
    # Normalize işlemi
    norm_accuracy = normalize(metrics["accuracy"], accuracy_min, accuracy_max)
    norm_f1 = normalize(metrics["f1_score"], f1_min, f1_max)
    norm_kfold = normalize(metrics["k_fold_mean"], kfold_min, kfold_max)
    
    # Toplam skor
    final_score = (
        weights["accuracy"] * norm_accuracy +
        weights["f1_score"] * norm_f1 +
        weights["kfold_mean"] * norm_kfold
    )
    final_scores[model] = final_score

# En iyi modeli bul
best_model = max(final_scores, key=final_scores.get)
print("\n\nEn iyi model:", best_model, "\n")

# Skorları DataFrame'e dönüştürme ve yazdırma
scores_df = pd.DataFrame.from_dict(final_scores, orient='index', columns=['Score'])
print(scores_df.sort_values(by='Score', ascending=False))
#========================================================================


#=============================Görselleştirme=============================
# Logistic Regression
x_set = ss.inverse_transform(x_train)
y_set = y_train

X1, X2 = np.meshgrid(
    np.arange(start=x_set[:, 0].min() - 10, stop=x_set[:, 0].max() + 10, step=1),
    np.arange(start=x_set[:, 1].min() - 1000, stop=x_set[:, 1].max() + 1000, step=1)
)

plt.contourf(X1, X2, lr_classifier.predict(ss.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
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


# Decision Tree
x_set_dt = ss.inverse_transform(x_test)
y_set_dt = y_test

X1, X2 = np.meshgrid(
    np.arange(start=x_set_dt[:, 0].min() - 1, stop=x_set_dt[:, 0].max() + 1, step=1),
    np.arange(start=x_set_dt[:, 1].min() - 1000, stop=x_set_dt[:, 1].max() + 1000, step=1)
)

plt.contourf(X1, X2, dt_classifier.predict(ss.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'blue')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set_dt)):
    plt.scatter(
        x_set_dt[y_set_dt == j, 0],
        x_set_dt[y_set_dt == j, 1],
        color=ListedColormap(('red', 'blue'))(i),
        label=j
    )

plt.title('Decision Tree - Eğitim Seti')
plt.xlabel('Yaş')
plt.ylabel('Maaş')
plt.legend()
plt.show()


# K-Neighbors
x_set_kn = ss.inverse_transform(x_test)
y_set_kn = y_test

X1, X2 = np.meshgrid(
    np.arange(start=x_set_kn[:, 0].min() - 1, stop=x_set_kn[:, 0].max() + 1, step=1),
    np.arange(start=x_set_kn[:, 1].min() - 1000, stop=x_set_kn[:, 1].max() + 1000, step=1)
)

plt.contourf(X1, X2, kn_classifier.predict(ss.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'blue')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set_kn)):
    plt.scatter(
        x_set_kn[y_set_kn == j, 0],
        x_set_kn[y_set_kn == j, 1],
        color=ListedColormap(('red', 'blue'))(i),
        label=j
    )

plt.title('K-Neighbors - Eğitim Seti')
plt.xlabel('Yaş')
plt.ylabel('Maaş')
plt.legend()
plt.show()


# Naive Bayes
x_set_nb = ss.inverse_transform(x_train)
y_set_nb = y_train

X1, X2 = np.meshgrid(
    np.arange(start=x_set_nb[:, 0].min() - 10, stop=x_set_nb[:, 0].max() + 10, step=1),
    np.arange(start=x_set_nb[:, 1].min() - 1000, stop=x_set_nb[:, 1].max() + 1000, step=1)
)

plt.contourf(X1, X2, nb_classifier.predict(ss.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'blue')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set_nb)):
    plt.scatter(
        x_set_nb[y_set_nb == j, 0],
        x_set_nb[y_set_nb == j, 1],
        color=ListedColormap(('red', 'blue'))(i),
        label=j
    )

plt.title('Naive Bayes - Eğitim Seti')
plt.xlabel('Yaş')
plt.ylabel('Maaş')
plt.legend()
plt.show()


# Support Vector Machine
x_set_svm = ss.inverse_transform(x_train)
y_set_svm = y_train

X1, X2 = np.meshgrid(
    np.arange(start=x_set_svm[:, 0].min() - 10, stop=x_set_svm[:, 0].max() + 10, step=1),
    np.arange(start=x_set_svm[:, 1].min() - 1000, stop=x_set_svm[:, 1].max() + 1000, step=1)
)

plt.contourf(X1, X2, svm_classifier.predict(ss.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'blue')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set_svm)):
    plt.scatter(
        x_set_svm[y_set_svm == j, 0],
        x_set_svm[y_set_svm == j, 1],
        color=ListedColormap(('red', 'blue'))(i),
        label=j
    )

plt.title('Support Vector Machine - Eğitim Seti')
plt.xlabel('Yaş')
plt.ylabel('Maaş')
plt.legend()
plt.show()
#========================================================================
