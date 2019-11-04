from sklearn.cluster import AgglomerativeClustering 
import scipy.cluster.hierarchy as shc 
import matplotlib.pyplot as plt 
import numpy as np 
 
X = np.array([[1, 2], [1, 4], [1, 0], 
			[4, 2], [4, 4], [4, 0]]) 

clustering = AgglomerativeClustering(n_clusters = 2).fit(X) 
plt.figure(figsize =(8, 8)) 
plt.title('Visualising the data') 
Dendrogram = shc.dendrogram((shc.linkage(X, method ='ward')))

print(clustering.labels_) 
