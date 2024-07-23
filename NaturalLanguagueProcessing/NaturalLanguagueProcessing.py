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

