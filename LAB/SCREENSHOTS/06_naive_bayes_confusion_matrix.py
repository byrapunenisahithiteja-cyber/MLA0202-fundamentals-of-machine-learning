import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, classification_report

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.25, random_state=42, stratify=data.target)
model = GaussianNB().fit(X_train, y_train)
pred = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, pred)*100, 2), "%")
print(classification_report(y_test, pred, target_names=data.target_names))
ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=data.target_names)
plt.title("Naive Bayes Confusion Matrix"); plt.tight_layout(); plt.show()
