
# Assignment 3: Unsupervised Learning + NLP (Bonus)
# Fill in the TODO sections and write short theory answers where asked.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA


# Load Dataset


# TODO: Load the Spotify dataset (or dataset used in Assignment 2)
df = pd.read_csv("dataset.csv")
df.head()

# Question 1: Data Preparation
# Q - Why is feature scaling important in clustering?

# Feature scaling is important because clustering algorithms use distance measures, and without scaling, features with larger values can dominate the results and distort the clusters.


features = ["danceability", "energy", "loudness", "tempo", "valence"]

# TODO: Select features
X = df[features]

# TODO: Handle missing values
print("Missing values:\n", X.isnull().sum())
X = X.dropna()
print("Shape after dropna:", X.shape)
 

scaler = StandardScaler()
# TODO: Scale features
X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(X_scaled, columns=features)
X_scaled.head()


# Question 2: Hierarchical Clustering
# Q: What does a dendrogram show?

# A dendrogram is a tree-like diagram that visualizes the hierarchical relationships and clustering structure among data points or objects



# TODO: Perform linkage
linked = linkage(X_scaled, method="ward")

 

plt.figure(figsize=(10,5))
dendrogram(linked)
plt.show()


 

# TODO: Choose number of clusters
n_clusters = 4

hc = AgglomerativeClustering(n_clusters=n_clusters)
df["HC_Cluster"] = hc.fit_predict(X_scaled)
df.head()

# Question 3: K-Means Clustering
# Q: Why does K-Means need K?

# K-Means clustering requires the parameter K because it is a partitioning algorithm that must know in advance how many clusters to form from the data.


 
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)


plt.plot(range(1,11), wcss, marker='o')
plt.show()


 

# TODO: Choose optimal K
optimal_k = 3

kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)
df.head()


# Question 4: PCA Visualization


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
X_pca = pd.DataFrame(X_pca, columns=["PC1","PC2"])


sns.scatterplot(x=X_pca.PC1, y=X_pca.PC2, hue=df["KMeans_Cluster"])
plt.show()


sns.scatterplot(x=X_pca.PC1, y=X_pca.PC2, hue=df["HC_Cluster"])
plt.show()


# # Question 5: Gaussian Mixture Models
# # Q: Difference between K-Means and GMM?

# K-Means and Gaussian Mixture Models (GMM) are both unsupervised clustering algorithms, but they differ fundamentally in their approach to grouping data points. K-Means uses hard assignments to cluster centers, while GMM employs probabilistic soft assignments based on Gaussian distributions.


# TODO: Initialize and fit GMM
gmm = GaussianMixture(n_components=optimal_k, random_state=42)
df["GMM_Cluster"] = gmm.fit_predict(X_scaled)
df.head()


# TODO: Show soft cluster probabilities
gmm_probs = gmm.predict_proba(X_scaled)
gmm_probs[:5]



# BONUS: NLP

  

text = "Machine learning enables systems to learn from data."
# TODO: Tokenization, stemming, lemmatization

import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# 1. Tokenization
tokens = word_tokenize(text)
print("Tokens:", tokens)

# 2. Stemming
stemmer = PorterStemmer()
stems = [stemmer.stem(word) for word in tokens]
print("Stems:", stems)

# 3. Lemmatization
lemmatizer = WordNetLemmatizer()
lemmas = [lemmatizer.lemmatize(word) for word in tokens]
print("Lemmas:", lemmas)
  

from sklearn.feature_extraction.text import CountVectorizer

sentences = [
    "Machine learning is powerful",
    "Unsupervised learning finds patterns",
    "Clustering groups similar data"
]

# TODO: Apply Bag of Words or TF-IDF

# Bag of Words using CountVectorizer
vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform(sentences)

# Display vocabulary and matrix
print("Vocabulary:", vectorizer.get_feature_names_out())
print("Bag of Words matrix:\n", bow_matrix.toarray())

# For TF-IDF, use TfidfVectorizer instead
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
print("\nTF-IDF matrix:\n", tfidf_matrix.toarray())

