import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

iris = load_iris()
X, y = iris.data, iris.target
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, random_state=42, stratify=y)
model = KNeighborsClassifier(n_neighbors=5).fit(Xtr, ytr)
pred = model.predict(Xte)
print("Accuracy:", round(accuracy_score(yte, pred)*100, 2), "%")
sample = [[5.1, 3.5, 1.4, .2]]
print("Predicted flower:", iris.target_names[model.predict(sample)[0]])
ConfusionMatrixDisplay.from_predictions(yte, pred, display_labels=iris.target_names)
plt.title("Iris Classification using K-NN"); plt.tight_layout(); plt.show()
