from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score 
from sklearn.preprocessing import StandardScaler
from data_loader import load_blobs
import matplotlib.pyplot as plt

X,y = load_blobs()

scaler = StandardScaler()

x_scaled = scaler.fit_transform(X)

mx_score =0

for eps in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
  for sample in [2,3,4,5,6,7,8,9,10]:
    model = DBSCAN(eps=eps,min_samples=sample)
    labels = model.fit_predict(x_scaled)
    num_cluster = len(set(labels)) - (1 if -1 in labels else 0)
    if num_cluster>=2:
      score = silhouette_score(x_scaled,labels)
      if score > mx_score:
        mx_score = score
        print(f"Best Score : eps({eps}) sample ({sample})")


model = DBSCAN(eps=0.3, min_samples= 6)
labels = model.fit_predict(x_scaled)

sscore = silhouette_score(x_scaled,labels)
ari = adjusted_rand_score(y, labels)
nmi = normalized_mutual_info_score(y, labels)

print( f"Silhouette score {sscore:.3f}")
print( f"ari score {ari:.3f}")
print( f"nmi score {nmi:.3f}")

plt.scatter(x_scaled[:,0],x_scaled[:,1], c=labels, cmap='viridis', marker='o')
plt.show()



