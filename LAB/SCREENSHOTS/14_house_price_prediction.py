import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Illustrative dataset: area in square feet, price in lakh units.
area = np.array([600,800,1000,1200,1400,1600,1800,2000]).reshape(-1,1)
price = np.array([18,23,29,34,40,45,51,57])
model = LinearRegression().fit(area, price)
pred = model.predict(area)
print("MAE:", round(mean_absolute_error(price, pred), 2), "lakh")
print("Estimated price for 1500 sq ft:", round(model.predict([[1500]])[0], 2), "lakh")
plt.scatter(area, price, label="Actual sample")
plt.plot(area, pred, label="Regression line")
plt.xlabel("Area (sq ft)"); plt.ylabel("Price (lakh)")
plt.title("House Price Prediction (illustrative data)"); plt.legend(); plt.grid(alpha=.3); plt.show()
