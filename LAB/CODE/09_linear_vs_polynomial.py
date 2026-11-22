import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

X = np.arange(1, 11).reshape(-1, 1)
y = np.array([3, 5, 8, 14, 22, 32, 45, 61, 80, 102])
linear = LinearRegression().fit(X, y)
poly = make_pipeline(PolynomialFeatures(degree=2), LinearRegression()).fit(X, y)
print("Linear MSE:", round(mean_squared_error(y, linear.predict(X)), 2))
print("Polynomial MSE:", round(mean_squared_error(y, poly.predict(X)), 2))
grid = np.linspace(1, 10, 200).reshape(-1, 1)
plt.scatter(X, y, label="Actual data")
plt.plot(grid, linear.predict(grid), label="Linear regression")
plt.plot(grid, poly.predict(grid), label="Polynomial regression (degree 2)")
plt.xlabel("X"); plt.ylabel("Y"); plt.title("Linear vs Polynomial Regression")
plt.legend(); plt.grid(alpha=.3); plt.show()
