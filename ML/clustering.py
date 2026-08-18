from data_loader import load_wine
from sklearn.cluster import KMeans, AgglomerativeClustering,DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score,normalized_mutual_info_score,adjusted_rand_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram,linkage
X,y = load_wine()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def print_metrics(labels):
   s_score = silhouette_score(X_scaled,labels)
   ari = adjusted_rand_score(y,labels)
   nmi = normalized_mutual_info_score(y,labels)

   print(f'silhouette score :{s_score:.3f}')
   print(f'ARI : {ari:.2f}')
   print(f'NMI: {nmi:2f}')

def show_data(method, labels):
  pca = PCA(
     n_components=2,
     random_state=42
  )
  xpca = pca.fit_transform(X_scaled)
  plt.scatter(xpca[:,0],xpca[:,1],c=labels,marker='o',cmap='viridis')
  plt.title(f'Visialization of {method}')
  plt.show()


def kmeans_cluster():
  
  xval = range(1,11)
  wcss =[]
  for k in xval:
    model = KMeans(k,random_state=42)
    model.fit(X_scaled)
    wcss.append(model.inertia_)
  
  plt.plot(xval,wcss)
  plt.title("Elbow Method to find Best K")
  plt.grid()
  plt.show()
  
  model = KMeans(n_clusters=3,random_state=42)
  labels = model.fit_predict(X_scaled)
  
  print_metrics(labels)
  show_data('KMeans',labels)
  

def hiearchical_cluster():
   
  z = linkage(X_scaled,method='ward')
  dendrogram(z)
  plt.title('Denrogram')
  plt.show()

  model = AgglomerativeClustering(n_clusters=3)
  labels = model.fit_predict(X_scaled)
  print_metrics(labels)
  show_data('Hearchichal',labels)


def dbscan_cluster():
   for eps in [0.2,0.3,0.4,0.5,0.7,0.8,0.9,1.0,1.5,2.0,2.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5]:
     for sample in [2,3,4,5,6,7,8,9]:
      model = DBSCAN(eps=eps,min_samples=sample)
      labels = model.fit_predict(X_scaled)
      n_cluster= len(set(labels)) - (1 if -1 in labels else 0)
      if n_cluster>=2:
        print_metrics(labels)
        print(f"EPS: {eps}, sample:{sample}")
   
   model = DBSCAN(eps=2.5,min_samples=8)
   labels = model.fit_predict(X_scaled)
   n_cluster= len(set(labels)) - (1 if -1 in labels else 0)
   if n_cluster>=2:
    print_metrics(labels)
    show_data('DBScan', labels)


def gmm():

  for k in range(1,11):
    model = GaussianMixture(n_components=k,random_state=42)
    model.fit(X_scaled)
    print(f"k = {k} , AIC{model.aic(X_scaled)}, BIC = {model.bic(X_scaled)}")
  
  model = GaussianMixture(n_components=3,random_state=42)
  
  labels = model.fit_predict(X_scaled)
  print_metrics(labels)
  show_data('GMM', labels)


if __name__ == "__main__":
    gmm()
