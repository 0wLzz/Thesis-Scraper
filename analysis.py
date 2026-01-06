import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from wordcloud import WordCloud, STOPWORDS
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

datasets = [
    "70-Outstanding-Graduates.csv",
    "72-Graduates.csv",
    "72-Outstanding-Graduates.csv",
    "73-Graduates.csv",
    "73-Outstanding-Graduates.csv",
]

@dataclass
class Configuration:
    max_features : int = 30
    ngrams_min : int = 1
    ngrams_max : int = 2
    cluster_k : int = 3
    min_df : int = 3
    max_df : float = 0.85

# Preprocessing
stopwords = set(STOPWORDS)
def remove_stopwords(title: str):
    return " ".join(word for word in title.split() if word not in stopwords)

def generate_wordcloud(titles : pd.Series):
    titles = titles.apply(remove_stopwords)
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(titles.to_string())
    
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title("Thesis Title Word Cloud")
    plt.axis("off")
    plt.show()

def generate_count(titles: pd.Series, cfg : Configuration):
    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(cfg.ngrams_min, cfg.ngrams_max),
        max_features= cfg.max_features
    )

    x = vectorizer.fit_transform(titles)
    freq = zip(vectorizer.get_feature_names_out(), x.sum(axis=0).tolist()[0])
    freq_df = pd.DataFrame(freq, columns=["term", "count"]).sort_values("count", ascending=False)
    
    return freq_df

def generate_cluster(titles: pd.Series, cfg: Configuration):
    tfidf = TfidfVectorizer(
        stop_words="english",
        max_df=cfg.max_df,
        min_df=cfg.min_df
    )

    x_tfidf = tfidf.fit_transform(titles)
    model = KMeans(n_clusters=cfg.cluster_k, random_state=42)
    model.fit(x_tfidf)

    terms = tfidf.get_feature_names_out()
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]

    for i in range(cfg.cluster_k):
        print(f"\nCluster {i}")
        print(", ".join([terms[ind] for ind in order_centroids[i, :10]]))

    return model.fit_predict(x_tfidf)

def load_dataset() -> list[pd.DataFrame]:
    dfs = []
    for dataset in datasets:
        df = pd.read_csv(os.path.join("data", dataset))
        df["Source"] = dataset
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)

def show_menu():
    print("\n=== Thesis Title Analysis Menu ===")
    print("1. Generate Word Cloud")
    print("2. Show Top Keywords")
    print("3. Run Clustering")
    print("4. Filter by Personal Keywords")
    print("0. Exit")


def menu():
    cfg = Configuration()
    data = load_dataset()
    data = data.drop_duplicates(subset=["Thesis Title"])
    titles = data["Thesis Title"]

    while True:
        show_menu()
        choice = input("Select option: ")

        if choice == "1":
            generate_wordcloud(titles)
        
        elif choice == "2":
            freq = generate_count(titles, cfg)
            print(freq)
        
        elif choice == "3":
            clusters = generate_cluster(titles, cfg)
            data["Cluster"] = clusters
            print(data[["Thesis Title", "Cluster"]].sample(10))

        elif choice == "4":
            keywords = input("Enter keywords (comma-separated): ").lower().split(",")
            mask = titles.str.lower().str.contains("|".join(k.strip() for k in keywords))

            filtered = data[mask]
            filtered.to_csv("Similar_Thesis_Title.csv", index=False)
            print(f"{len(filtered)} titles saved.")

        elif choice == "0":
            print("Exiting.")
            break

        else:
            print("Invalid option.")


if __name__ == "__main__":
    menu()