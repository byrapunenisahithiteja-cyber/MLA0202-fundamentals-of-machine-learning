import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

iris = load_iris()
Xtr, Xte, ytr, yte = train_test_split(iris.data, iris.target, test_size=.25,
                                       random_state=42, stratify=iris.target)
models = {
 "K-NN": make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
 "Decision Tree": DecisionTreeClassifier(random_state=42),
 "Naive Bayes": GaussianNB(),
 "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
}
scores = {}
for name, model in models.items():
    model.fit(Xtr, ytr)
    scores[name] = accuracy_score(yte, model.predict(Xte))*100
    print(name, "accuracy:", round(scores[name], 2), "%")
plt.bar(scores.keys(), scores.values())
plt.ylabel("Accuracy (%)"); plt.ylim(0, 105)
plt.title("Classification Algorithm Comparison")
plt.xticks(rotation=15); plt.grid(axis="y", alpha=.3); plt.tight_layout(); plt.show()
