import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data[:, :2]  # sepal length and width for 2D plotting
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, pred)*100, 2), "%")
new = [[5.1, 3.5]]
print("New sample predicted class:", iris.target_names[model.predict(new)[0]])
plt.scatter(X[:,0], X[:,1], c=y, alpha=.65, label="Iris samples")
plt.scatter(new[0][0], new[0][1], marker="*", s=250, label="New sample")
plt.xlabel("Sepal length (cm)"); plt.ylabel("Sepal width (cm)")
plt.title("K-NN on Iris Dataset"); plt.legend(); plt.grid(alpha=.25); plt.show()
