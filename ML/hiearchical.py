from data_loader import load_blobs
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
X,y = load_blobs()

scaler = StandardScaler()
X = scaler.fit_transform(X)

Z = linkage(X, method='ward')

dendrogram(Z)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.grid()

plt.show()


model = AgglomerativeClustering(n_clusters=4)
model.fit(X)

labels = model.labels_

plt.scatter(X[:,0],X[:,1],c=labels,cmap='viridis', marker='o')
plt.title("Heiarchical Clustering")
plt.show()

sil_score = silhouette_score(X,labels)
ari = adjusted_rand_score(y,labels)
nmi = normalized_mutual_info_score(y,labels)
print(f"Silhouette Score {sil_score:.3f}")
print(f"ari Score {ari:.3f}")
print(f"nmi Score {nmi:.3f}")