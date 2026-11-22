import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Illustrative data: age in years and price in thousands of currency units.
age = np.array([1,2,3,4,5,6,7,8]).reshape(-1,1)
price = np.array([950,850,760,650,570,480,390,320])
model = LinearRegression().fit(age, price)
pred = model.predict(age)
print("MAE:", round(mean_absolute_error(price, pred), 2))
print("Estimated price for a 3-year-old car:", round(model.predict([[3]])[0], 2), "(thousand units)")
plt.scatter(age, price, label="Sample prices")
plt.plot(age, pred, label="Linear model")
plt.xlabel("Car age (years)"); plt.ylabel("Price (thousands)")
plt.title("Car Price Prediction (illustrative data)"); plt.legend(); plt.grid(alpha=.3); plt.show()
