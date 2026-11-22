import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

# XOR dataset
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])
model = MLPClassifier(hidden_layer_sizes=(4,), activation="tanh",
                      solver="lbfgs", max_iter=5000, random_state=7)
model.fit(X, y)
pred = model.predict(X)
print("Input | Actual | Predicted")
for a, b, c in zip(X, y, pred): print(a, "|", b, "|", c)
print("Training accuracy:", model.score(X, y) * 100, "%")
plt.scatter(X[:,0], X[:,1], c=pred, s=220)
for i, p in enumerate(pred): plt.annotate(f"Output={p}", X[i] + 0.03)
plt.xlabel("Input 1"); plt.ylabel("Input 2"); plt.title("ANN Backpropagation: XOR")
plt.xticks([0,1]); plt.yticks([0,1]); plt.grid(alpha=.3); plt.show()
