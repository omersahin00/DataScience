import pandas as pdimport numpy as npimport matplotlib.pyplot as pltdataset = pd.read_csv("breast_cancer.csv")x = dataset.iloc[:, 1:-1].valuesy = dataset.iloc[:, -1].valuesfrom sklearn.model_selection import train_test_splitx_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)from sklearn.preprocessing import StandardScalerss = StandardScaler()x_train[:, 3:] = ss.fit_transform(x_train[:, 3:])x_test[: , 3:] = ss.transform(x_test[:, 3:])from sklearn.svm import SVCclassifier = SVC(probability=True, kernel="rbf")classifier.fit(x_train, y_train)y_pred = classifier.predict(x_test)from sklearn.metrics import confusion_matrix, accuracy_scorecmSVM = confusion_matrix(y_test, y_pred)AccuracyScoreSVM = accuracy_score(y_test, y_pred)#k-Fold Cross Validation yöntemiyle modelin performansının ölçülmesifrom sklearn.model_selection import cross_val_scoreaccuriciesSVM = cross_val_score(estimator = classifier, X = x, y = y, cv = 10)AccuriciesMeanSVM = accuriciesSVM.mean() * 100StadartDeviationSVM = accuriciesSVM.std() * 100