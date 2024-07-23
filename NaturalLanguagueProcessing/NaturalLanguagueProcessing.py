import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Restaurant_Reviews.tsv", sep="\t", quoting=3)

#Cleaning the Text
import re

print(dataset["Review"][0])
print(re.sub("[^a-z A-Z]", "", dataset["Review"][0]))

for i in range(0, 1000):
    cleanText = re.sub("[^a-z A-Z]", "", dataset["Review"][i])
    cleanText = cleanText.upper().lower()
    cleanText = cleanText.split()
    dataset["Review"][i] = cleanText


#Stopwords
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords

all_stopwords = stopwords.words("english")
all_stopwords.remove("not")

for i in range(0, 1000):
    temp_list = []
    for word in dataset["Review"][i]:
        if word in all_stopwords:
            continue
        else:
            temp_list.append(word)
    dataset["Review"][i] = temp_list


#Stemming
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


for i in range(0, 1000):
    temp_list = []
    for word in dataset["Review"][i]:
        temp_list.append(ps.stem(word))
    dataset["Review"][i] = temp_list


#Cümleleri geri birleştirme
for i in range(0, 1000):
    dataset["Review"][i] = " ".join(dataset["Review"][i])


#Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)

x = cv.fit_transform(dataset["Review"]).toarray()
y = dataset.iloc[:, -1].values


#Train ve Test Setleri
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)


# TAHMİNLER:

#Naive Bayes Modeli ile Tahmin      BAŞARISI: 0.69
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()

classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score

cmNB = confusion_matrix(y_test, y_pred)
AccuracyScoreNB = accuracy_score(y_test, y_pred)



#SVM Modeli ile Tahmin             BAŞARISI: 0.825
from sklearn.svm import SVC
classifier = SVC(probability = True, kernel = "rbf")

classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cmSVM = confusion_matrix(y_test, y_pred)
AccuracyScoreSVM = accuracy_score(y_test, y_pred)









