# Install required libraries
!pip install scikit-learn pandas numpy

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Dataset (IMDb reviews sample from sklearn or replace with your dataset)
from sklearn.datasets import fetch_20newsgroups

# For simplicity, let's take a binary subset (rec.sport.hockey vs sci.med)
categories = ['rec.sport.hockey', 'sci.med']
data = fetch_20newsgroups(subset='all', categories=categories, remove=('headers','footers','quotes'))

df = pd.DataFrame({'text': data.data, 'label': data.target})
print(df.head())

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# -------- BAG OF WORDS --------
bow_vectorizer = CountVectorizer(stop_words='english', max_features=5000)
X_train_bow = bow_vectorizer.fit_transform(X_train)
X_test_bow = bow_vectorizer.transform(X_test)

model_bow = LogisticRegression(max_iter=1000)
model_bow.fit(X_train_bow, y_train)
y_pred_bow = model_bow.predict(X_test_bow)

print("\n--- Bag of Words Results ---")
print("Accuracy:", accuracy_score(y_test, y_pred_bow))
print(classification_report(y_test, y_pred_bow))

# -------- TF-IDF --------
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

model_tfidf = LogisticRegression(max_iter=1000)
model_tfidf.fit(X_train_tfidf, y_train)
y_pred_tfidf = model_tfidf.predict(X_test_tfidf)

print("\n--- TF-IDF Results ---")
print("Accuracy:", accuracy_score(y_test, y_pred_tfidf))
print(classification_report(y_test, y_pred_tfidf))