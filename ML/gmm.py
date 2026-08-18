from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
from data_loader import load_blobs
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X,y = load_blobs()

x_scaled = scaler.fit_transform(X)

for k in range(1, 11):
  model = GaussianMixture(n_components=k,random_state=42)
  model.fit(x_scaled)
  print(f"K = {k}",f"AIC = {model.aic(x_scaled)}",f"BIC = {model.bic(x_scaled)}")


model = GaussianMixture(n_components=4,random_state=42)

labels = model.fit_predict(x_scaled)

plt.scatter(x_scaled[:,0],x_scaled[:,1], c=labels, cmap='viridis',marker='o')
plt.show()