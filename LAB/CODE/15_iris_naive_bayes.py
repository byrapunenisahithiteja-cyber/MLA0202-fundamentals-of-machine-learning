import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

iris = load_iris()
Xtr, Xte, ytr, yte = train_test_split(iris.data, iris.target, test_size=.25,
                                       random_state=42, stratify=iris.target)
model = GaussianNB().fit(Xtr, ytr)
pred = model.predict(Xte)
print("Accuracy:", round(accuracy_score(yte, pred)*100, 2), "%")
print("Predicted class for sample:", iris.target_names[model.predict([[5.1,3.5,1.4,.2]])[0]])
ConfusionMatrixDisplay.from_predictions(yte, pred, display_labels=iris.target_names)
plt.title("Iris Classification using Naive Bayes"); plt.tight_layout(); plt.show()
