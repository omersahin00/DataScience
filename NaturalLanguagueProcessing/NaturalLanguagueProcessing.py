import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Veri setini yükleme
dataset = pd.read_csv("Restaurant_Reviews.tsv", sep="\t", quoting=3)

# Gerekli NLTK Verilerini İndirme
nltk.download("stopwords")

# Metin Temizleme ve Ön İşleme
ps = PorterStemmer()
all_stopwords = stopwords.words("english")
all_stopwords.remove("not")

def clean_text(text):
    # Sadece harf karakterlerini bırak
    text = re.sub("[^a-zA-Z]", " ", text)
    # Küçük harfe dönüştür
    text = text.lower()
    # Kelimelere ayır ve stopwords ile stem işlemi uygula
    words = [ps.stem(word) for word in text.split() if word not in all_stopwords]
    return " ".join(words)

# Tüm yorumlara temizleme işlemini uygula
dataset["Cleaned_Review"] = dataset["Review"].apply(clean_text)

# Özellik ve hedef değişkenleri ayırma
cv = CountVectorizer(max_features=1500)
x = cv.fit_transform(dataset["Cleaned_Review"]).toarray()
y = dataset["Liked"].values

# Eğitim ve Test Verilerini Bölme
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Naive Bayes Modeli
nb_classifier = GaussianNB()
nb_classifier.fit(x_train, y_train)
y_pred_nb = nb_classifier.predict(x_test)

# Naive Bayes Performans Sonuçları
print("Naive Bayes Results:")
print(confusion_matrix(y_test, y_pred_nb))
print(f"Accuracy: {accuracy_score(y_test, y_pred_nb):.2f}")
print(classification_report(y_test, y_pred_nb))

# SVM Modeli
svm_classifier = SVC(probability=True, kernel="rbf", random_state=1)
svm_classifier.fit(x_train, y_train)
y_pred_svm = svm_classifier.predict(x_test)

# SVM Performans Sonuçları
print("\nSVM Results:")
print(confusion_matrix(y_test, y_pred_svm))
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.2f}")
print(classification_report(y_test, y_pred_svm))
