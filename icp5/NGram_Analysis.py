from sklearn.feature_extraction.text import CountVectorizer

from Word2Vec import spark

vectorizer=CountVectorizer()
data_corpus=["guru99 is the best sitefor online tutorials. I love to visit guru99."]
vocabulary=vectorizer.fit(data_corpus)
X= vectorizer.transform(data_corpus)
# vocabulary=vectorizer.fit(documentDF)
# X= vectorizer.transform(documentDF)
print(X.toarray())
print(vocabulary.get_feature_names())