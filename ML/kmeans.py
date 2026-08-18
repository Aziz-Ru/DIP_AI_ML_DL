from data_loader import load_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

X,y = load_blobs()

# Elbow Method to find the optimal number of clusters
num_clusters = range(1, 11)
wcss = []

for k in num_clusters:
  kmeans = KMeans(n_clusters=k, random_state=42)
  kmeans.fit(X)
  wcss.append(kmeans.inertia_)

plt.plot(num_clusters, wcss, marker='o')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
# plt.xticks(num_clusters)
plt.grid()
plt.show()

model = KMeans(n_clusters=4, random_state=42)
model.fit(X)
labels = model.labels_
print(labels)
s_score = silhouette_score(X, labels)
ari = adjusted_rand_score(y,labels)
nmi = normalized_mutual_info_score(y,labels)
print(f"Silhouette Score: {s_score}")
print(f"ARI: {ari:.2f}")
print(f"NMI {nmi:.2f}")
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k')
plt.title('K-Means Clustering Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid()
plt.show()





  
