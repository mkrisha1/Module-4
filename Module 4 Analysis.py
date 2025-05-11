# Module 4 Analysis 

# Imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Step 1: Load Dataset
df = pd.read_csv("spotify-2023.csv", encoding="latin1")
df = df[['track_name', 'artist(s)_name', 'danceability_%', 'energy_%', 'speechiness_%']]  # Rename columns if needed

# Step 2: Clean & Prepare Data
# Rename columns for consistency
df.rename(columns={
    'danceability_%': 'danceability',
    'energy_%': 'energy',
    'speechiness_%': 'speechiness'
}, inplace=True)

# Drop missing values
df.dropna(subset=['danceability', 'energy', 'speechiness'], inplace=True)

# Step 3: Normalize Selected Features
features = df[['danceability', 'energy', 'speechiness']]
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Step 4: Determine Optimal k (Elbow Method)
inertias = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(scaled_features)
    inertias.append(model.inertia_)
    silhouette_scores.append(silhouette_score(scaled_features, model.labels_))

# Plot Elbow Curve
plt.figure(figsize=(8, 4))
plt.plot(k_range, inertias, marker='o')
plt.title('Elbow Method: Finding the Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.xticks(k_range)
plt.grid(True)
plt.show()

# Step 5: Final Clustering with Optimal k
optimal_k = 4  # Replace with value from elbow/silhouette
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['cluster'] = kmeans.fit_predict(scaled_features)

# Step 6: Analyze Clusters
for i in range(optimal_k):
    cluster_df = df[df['cluster'] == i]
    print(f"\nCluster {i}:")
    print(cluster_df[['track_name', 'artist(s)_name']].head(3))  # Show 3 example songs
    print(cluster_df[['danceability', 'energy', 'speechiness']].mean())

# Step 7: Visualize Clusters (Danceability and Energy)
plt.figure(figsize=(8, 6))
plt.scatter(scaled_features[:, 0], scaled_features[:, 1], c=df['cluster'], cmap='viridis')
plt.title("Spotify Songs Clustering (Danceability vs. Energy)")
plt.xlabel("Danceability (scaled)")
plt.ylabel("Energy (scaled)")
plt.grid(True)
plt.colorbar(label='Cluster')
plt.show()

