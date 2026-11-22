import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

X, _ = make_blobs(n_samples=250, centers=3, cluster_std=1.2, random_state=42)
model = GaussianMixture(n_components=3, random_state=42).fit(X)
labels = model.predict(X)
print("Gaussian Mixture Model fitted using EM.")
print("Cluster counts:", __import__("numpy").bincount(labels))
plt.scatter(X[:,0], X[:,1], c=labels, s=35)
plt.scatter(model.means_[:,0], model.means_[:,1], marker="X", s=220, label="Cluster means")
plt.xlabel("Feature 1"); plt.ylabel("Feature 2")
plt.title("Expectation-Maximization (Gaussian Mixture)"); plt.legend(); plt.grid(alpha=.2); plt.show()
