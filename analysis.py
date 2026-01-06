import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re

paths = [
    "70-Outstanding-Graduates.csv",
    "72-Graduates.csv",
    "72-Outstanding-Graduates.csv",
    "73-Graduates.csv",
    "73-Outstanding-Graduates.csv",
]

# Preprocessing
stopwords = set(STOPWORDS)
def remove_stopwords(title: str):
    return " ".join(word for word in title.split() if word not in stopwords)

def generate_wordcloud(titles : pd.Series):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(titles.to_string())
    
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title("Thesis Title Word Cloud")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    dfs = []
    for path in paths:
        df = pd.read_csv(path)
        df["Source"] = path
        dfs.append(df)
    
    data = pd.concat(dfs, ignore_index=True)

    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=30
    )

    x = vectorizer.fit_transform(data["Thesis Title"])
    freq = zip(vectorizer.get_feature_names_out(), x.sum(axis=0).tolist()[0])
    freq_df = pd.DataFrame(freq, columns=["term", "count"]).sort_values("count", ascending=False)
    print(freq_df)

    tfidf = TfidfVectorizer(
        stop_words="english",
        max_df=0.85,
        min_df=3
    )

    X_tfidf = tfidf.fit_transform(data["Thesis Title"])

    k = 3  # play with this
    model = KMeans(n_clusters=k, random_state=42)
    data["cluster"] = model.fit_predict(X_tfidf)
    terms = tfidf.get_feature_names_out()
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]

    for i in range(k):
        print(f"\nCluster {i}")
        print(", ".join([terms[ind] for ind in order_centroids[i, :10]]))
    
    data["cluster"].value_counts()

    keywords = ["backend", "ai", "machine"]
    mask = data["Thesis Title"].str.lower().str.contains("|".join(keywords))
    personal_slice = data[mask]
    personal_slice.sample(10)
    personal_slice.to_csv("Recomended_Thesis_Title.csv")
    print(personal_slice)


