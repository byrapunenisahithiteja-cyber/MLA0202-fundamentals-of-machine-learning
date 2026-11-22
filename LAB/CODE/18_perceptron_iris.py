import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

iris = load_iris()
# Binary classification makes the classic perceptron task straightforward.
mask = iris.target < 2
X, y = iris.data[mask, :2], iris.target[mask]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, random_state=42, stratify=y)
model = make_pipeline(StandardScaler(), Perceptron(max_iter=2000, random_state=42))
model.fit(Xtr, ytr)
pred = model.predict(Xte)
print("Classes:", iris.target_names[:2])
print("Accuracy:", round(accuracy_score(yte, pred)*100, 2), "%")
ConfusionMatrixDisplay.from_predictions(yte, pred, display_labels=iris.target_names[:2])
plt.title("Perceptron Iris Classification (Setosa vs Versicolor)")
plt.tight_layout(); plt.show()
