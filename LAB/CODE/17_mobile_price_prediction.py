import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

# Small illustrative dataset; use the full Mobile Price Classification CSV for real results.
df = pd.DataFrame({
 "RAM":[2,2,3,3,4,4,6,6,8,8,12,12],
 "Battery":[2500,3000,2800,3500,3200,4000,3500,4500,4000,5000,4500,5000],
 "Storage":[16,32,32,64,64,128,128,128,256,256,256,512],
 "PriceRange":[0,0,1,1,1,2,2,2,3,3,3,3]
})
X, y = df.drop(columns="PriceRange"), df["PriceRange"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=100, random_state=42).fit(Xtr, ytr)
pred = model.predict(Xte)
print("Accuracy on illustrative test set:", round(accuracy_score(yte, pred)*100, 2), "%")
print("New phone predicted price range:", model.predict([[8,4500,256]])[0])
ConfusionMatrixDisplay.from_predictions(yte, pred)
plt.title("Mobile Price Range Classification"); plt.tight_layout(); plt.show()
