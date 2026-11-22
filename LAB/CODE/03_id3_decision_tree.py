import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import OrdinalEncoder

data = pd.DataFrame({
    "Outlook": ["Sunny","Sunny","Overcast","Rain","Rain","Rain","Overcast","Sunny","Sunny","Rain","Sunny","Overcast","Overcast","Rain"],
    "Temperature": ["Hot","Hot","Hot","Mild","Cool","Cool","Cool","Mild","Cool","Mild","Mild","Mild","Hot","Mild"],
    "Humidity": ["High","High","High","High","Normal","Normal","Normal","High","Normal","Normal","Normal","High","Normal","High"],
    "Wind": ["Weak","Strong","Weak","Weak","Weak","Strong","Strong","Weak","Weak","Weak","Strong","Strong","Weak","Strong"],
    "Play": ["No","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No"]
})
Xraw, y = data.drop(columns="Play"), data["Play"]
enc = OrdinalEncoder()
X = enc.fit_transform(Xraw)
model = DecisionTreeClassifier(criterion="entropy", random_state=0)
model.fit(X, y)
sample = pd.DataFrame([["Sunny","Cool","High","Strong"]], columns=Xraw.columns)
prediction = model.predict(enc.transform(sample))
print("New sample:", sample.to_dict(orient="records")[0])
print("Predicted class:", prediction[0])
plt.figure(figsize=(14, 7))
plot_tree(model, feature_names=Xraw.columns, class_names=model.classes_, filled=True, rounded=True)
plt.title("Decision Tree (Entropy / ID3-style)")
plt.tight_layout(); plt.show()
