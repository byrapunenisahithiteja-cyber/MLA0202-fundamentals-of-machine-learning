import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Illustrative monthly sales; replace with actual sales history for meaningful forecasts.
months = np.arange(1, 13).reshape(-1, 1)
sales = np.array([120,128,125,140,150,155,162,170,168,185,195,210])
model = LinearRegression().fit(months, sales)
future_months = np.arange(1, 16).reshape(-1, 1)
pred = model.predict(future_months)
print("MAE on training data:", round(mean_absolute_error(sales, model.predict(months)), 2))
print("Predicted sales for months 13, 14, 15:", np.round(pred[-3:], 1))
plt.plot(months, sales, "o-", label="Historical sales")
plt.plot(future_months, pred, "--", label="Linear trend / forecast")
plt.axvline(12.5, linestyle=":", label="Forecast begins")
plt.xlabel("Month"); plt.ylabel("Sales")
plt.title("Future Sales Prediction (illustrative data)")
plt.legend(); plt.grid(alpha=.3); plt.show()
