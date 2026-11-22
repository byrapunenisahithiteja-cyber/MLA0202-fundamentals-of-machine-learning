import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

# Illustrative categorical loan dataset; replace with institution-approved data.
df = pd.DataFrame({
 "Employment":["Salaried","Self-employed","Salaried","Unemployed","Salaried","Self-employed","Unemployed","Salaried","Self-employed","Salaried","Unemployed","Salaried"],
 "CreditHistory":["Good","Good","Poor","Poor","Good","Poor","Poor","Good","Good","Poor","Good","Good"],
 "IncomeLevel":["High","High","Medium","Low","Medium","Medium","Low","High","Medium","Low","Medium","High"],
 "LoanApproved":["Yes","Yes","No","No","Yes","No","No","Yes","Yes","No","Yes","Yes"]
})
X, y = df.drop(columns="LoanApproved"), df["LoanApproved"]
enc = OneHotEncoder(handle_unknown="ignore")
Xenc = enc.fit_transform(X).toarray()
# Convert one-hot features to categorical integer values for CategoricalNB.
Xtr, Xte, ytr, yte = train_test_split(Xenc, y, test_size=.25, random_state=42, stratify=y)
model = CategoricalNB().fit(Xtr.astype(int), ytr)
pred = model.predict(Xte.astype(int))
print("Accuracy on illustrative test set:", round(accuracy_score(yte, pred)*100, 2), "%")
sample = pd.DataFrame([{"Employment":"Salaried","CreditHistory":"Good","IncomeLevel":"High"}])
print("Loan decision for sample:", model.predict(enc.transform(sample).toarray().astype(int))[0])
ConfusionMatrixDisplay.from_predictions(yte, pred, labels=["No","Yes"])
plt.title("Bank Loan Prediction using Naive Bayes"); plt.tight_layout(); plt.show()
