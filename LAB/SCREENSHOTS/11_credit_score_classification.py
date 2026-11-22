import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

# Small illustrative dataset; replace with a real CSV for a real credit model.
df = pd.DataFrame({
 "Income":[25000,30000,45000,50000,70000,80000,35000,90000,42000,65000,28000,75000],
 "Debt":[18000,22000,12000,15000,10000,12000,26000,9000,16000,18000,21000,11000],
 "PaymentHistory":["Poor","Fair","Good","Good","Good","Good","Poor","Good","Fair","Good","Poor","Good"],
 "Employment":["Short","Short","Long","Long","Long","Long","Short","Long","Short","Long","Short","Long"],
 "Score":["Low","Low","Medium","Medium","High","High","Low","High","Medium","High","Low","High"]
})
X, y = df.drop(columns="Score"), df["Score"]
cat = ["PaymentHistory","Employment"]; num = ["Income","Debt"]
prep = ColumnTransformer([("num", make_pipeline(SimpleImputer(), StandardScaler()), num),
                          ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), cat)])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, random_state=2, stratify=y)
model = make_pipeline(prep, RandomForestClassifier(n_estimators=100, random_state=2))
model.fit(Xtr, ytr); pred = model.predict(Xte)
print("Accuracy on small illustrative test set:", round(accuracy_score(yte, pred)*100, 2), "%")
sample = pd.DataFrame([{"Income":60000,"Debt":14000,"PaymentHistory":"Good","Employment":"Long"}])
print("New applicant predicted score:", model.predict(sample)[0])
ConfusionMatrixDisplay.from_predictions(yte, pred, labels=["Low","Medium","High"])
plt.title("Credit Score Classification"); plt.tight_layout(); plt.show()
