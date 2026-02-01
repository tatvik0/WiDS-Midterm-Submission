#FINAL PROJECT:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# Part A: Data Loading and Understanding

# Load the Spotify dataset using Pandas

df = pd.read_csv("dataset.csv")

# Display dataset shape, columns, and data types

print("Dataset Shape:", df.shape)
print("Dataset Columns:", df.columns.tolist())

# Identify numerical and categorical features

numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = df.select_dtypes(exclude=[np.number]).columns.tolist()

print("Numerical Features:", numerical_features)
print("Categorical Features:", categorical_features)



# Part B: Probability and Statistics Analysis

# Compute mean, variance, and standard deviation of key audio features

mean_key_features = df[numerical_features].mean()
variance_key_features = df[numerical_features].var()

print("Mean of Key Audio Features:\n", mean_key_features)
print("Variance of Key Audio Features:\n", variance_key_features)

# Analyze distributions of energy, danceability, and tempo

features = ["energy", "danceability", "tempo"]
for feature in features:
    plt.hist(df[feature], bins=50,edgecolor='black')
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.show()

# Comment on observed statistical patterns

# Danceability and energy show moderate variability, indicating diversity in
# rhythmic and intensity characteristics of songs. Tempo shows higher variation,
# suggesting a wide range of song speeds in the dataset.



# Part C: Exploratory Data Analysis (EDA)

# Handle missing values (if any)

df.dropna(inplace=True)

# Perform feature scaling where required

from sklearn.preprocessing import StandardScaler

features_to_scale = ['danceability', 'energy', 'popularity','loudness', 'speechiness',
                     'acousticness', 'instrumentalness', 'liveness',
                     'valence', 'tempo','time_signature']

scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[features_to_scale] = scaler.fit_transform(df[features_to_scale])

# Create visualizations using Matplotlib / Seaborn:

# Feature distributions


# Correlation heatmap

correlation_matrix = df[features_to_scale].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            fmt='.2f', square=True, linewidths=0.5)
plt.title('Correlation Heatmap of Audio Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# Feature comparison across genres or moods

genres = df['track_genre'].value_counts().head(10).index
df_top = df[df['track_genre'].isin(genres)]

plt.figure(figsize=(12, 6))
sns.boxplot(data=df_top, x='track_genre', y='energy', color='lightblue')
plt.title('Energy Levels Across Genres ', fontsize=14, fontweight='bold')
plt.xlabel('Genre')
plt.ylabel('Energy')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()



# Part D: Supervised Learning


# Define a classification task (e.g., genre or mood prediction)

genres = df['track_genre'].value_counts().head(114).index
df_classification = df[df['track_genre'].isin(genres)].copy()
print(f"\nFiltered dataset size: {df_classification.shape[0]} songs")
print(f"Genres to classify: {genres.tolist()}")

X = df_classification[features_to_scale]
y = df_classification['track_genre']


# Split data into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set: {X_train.shape[0]} songs")
print(f"Testing set: {X_test.shape[0]} songs")

# Scale features

scaler_model = StandardScaler()
X_train_scaled = scaler_model.fit_transform(X_train)
X_test_scaled = scaler_model.transform(X_test)



# Train at least two supervised models from the following:

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# Logistic Regression

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)

y_pred_lr = lr_model.predict(X_test_scaled)

accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"Accuracy: {accuracy_lr:.4f} ({accuracy_lr*100:.2f}%)")


# K‑Nearest Neighbors (KNN)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

y_pred_knn = knn_model.predict(X_test_scaled)

accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"Accuracy: {accuracy_knn:.4f} ({accuracy_knn*100:.2f}%)")


# Decision Tree

dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train_scaled, y_train)

y_pred_dt = dt_model.predict(X_test_scaled)

accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"Accuracy: {accuracy_dt:.4f} ({accuracy_dt*100:.2f}%)")


# Evaluate models using:

# Accuracy

print("\nMODEL COMPARISON ")
models = ['Logistic Regression', 'K-Nearest Neighbors', 'Decision Tree']
accuracies = [accuracy_lr, accuracy_knn, accuracy_dt]

plt.figure(figsize=(10, 6))
bars = plt.bar(models, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
plt.ylabel('Accuracy', fontsize=12)
plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
plt.ylim([0, 1])
plt.grid(True, alpha=0.3, axis='y')

best_model_idx = np.argmax(accuracies)
best_model_name = models[best_model_idx]
best_predictions = [y_pred_lr, y_pred_knn, y_pred_dt][best_model_idx]
best_model = [lr_model, knn_model, dt_model][best_model_idx]

print(f"\nBest performing model: {best_model_name}")
print(f"Best accuracy: {accuracies[best_model_idx]:.4f}")

# Confusion Matrix

print(f"\nGenerating confusion matrix for {best_model_name}...")
cm = confusion_matrix(y_test, best_predictions)

# A larger heatmap for all genres

plt.figure(figsize=(20, 18))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
            xticklabels=sorted(y_test.unique()),
            yticklabels=sorted(y_test.unique()),
            cbar_kws={'label': 'Count'})
plt.title(f'Confusion Matrix - {best_model_name} (All Genres)',
          fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Actual Genre', fontsize=12)
plt.xlabel('Predicted Genre', fontsize=12)
plt.xticks(rotation=90, fontsize=6)
plt.yticks(rotation=0, fontsize=6)
plt.tight_layout()
plt.show()



# Part E: Model Testing on New Song Sample

print("\nTesting model with a new song sample...")

#Enter audio features of your song

new_song_features = {
    'danceability': 0.0,
    'energy': 0.0,
    'popularity': 0.0,
    'loudness': 0.0,
    'speechiness': 0.0,
    'acousticness':  0.0,
    'instrumentalness': 0.0,
    'liveness': 0.0,
    'valence': 0.0,
    'tempo': 0.0,
    'time_signature':0.0
}

print("\nAudio Features:")
for feature, value in new_song_features.items():
    print(f"  {feature:20s}: {value}")

# Input the features into the trained model
new_song_df = pd.DataFrame([new_song_features])

# Scale features
new_song_scaled = scaler_model.transform(new_song_df)


print("\nPREDICTIONS FROM ALL MODELS:")

pred_lr = lr_model.predict(new_song_scaled)[0]
pred_knn = knn_model.predict(new_song_scaled)[0]
pred_dt = dt_model.predict(new_song_scaled)[0]

print(f"\nLogistic Regression prediction: {pred_lr}")
print(f"K-Nearest Neighbors prediction: {pred_knn}")
print(f"Decision Tree prediction: {pred_dt}")


print(f"\nThe predicted genre of this song is:")
print(f"  → {pred_knn}")
