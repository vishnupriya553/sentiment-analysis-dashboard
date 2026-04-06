import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv("data.csv")

X = df["Text"]
y = df["Sentiment"]

# Convert text to numbers
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train model
model = MultinomialNB()
model.fit(X_vec, y)

def predict_sentiment(text):
    text_vec = vectorizer.transform([text])
    return model.predict(text_vec)[0]