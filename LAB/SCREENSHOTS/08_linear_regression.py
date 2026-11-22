import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X = np.array([[1],[2],[3],[4],[5],[6],[7],[8]])
y = np.array([15,18,25,29,35,39,45,49])
model = LinearRegression().fit(X, y)
pred = model.predict(X)
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("MSE:", round(mean_squared_error(y, pred), 3))
print("R2 score:", round(r2_score(y, pred), 3))
print("Prediction for X=9:", round(model.predict([[9]])[0], 2))
plt.scatter(X, y, label="Actual data")
plt.plot(X, pred, label="Regression line")
plt.xlabel("Study hours"); plt.ylabel("Score")
plt.title("Linear Regression"); plt.legend(); plt.grid(alpha=.3); plt.show()
