import pandas as pd
import matplotlib.pyplot as plt

# Example data; also saved as training_data.csv
df = pd.DataFrame({
    "Sky": ["Sunny", "Sunny", "Rainy", "Sunny"],
    "Temperature": ["Warm", "Warm", "Cold", "Warm"],
    "Humidity": ["Normal", "High", "High", "High"],
    "Wind": ["Strong", "Strong", "Strong", "Weak"],
    "EnjoySport": ["Yes", "Yes", "No", "Yes"]
})
df.to_csv("training_data.csv", index=False)
attrs = list(df.columns[:-1])
S = [None] * len(attrs)
G = [["?"] * len(attrs)]

def covers(h, x):
    return all(a == "?" or a == b for a, b in zip(h, x))

def min_generalizations(s, x):
    result = s.copy()
    for i, (a, b) in enumerate(zip(s, x)):
        if a is None: result[i] = b
        elif a != b: result[i] = "?"
    return result

def min_specializations(g, x, domains, s):
    out = []
    for i, val in enumerate(g):
        if val == "?":
            for d in domains[i]:
                if d != x[i] and (s[i] is None or s[i] == "?" or s[i] == d):
                    new = g.copy(); new[i] = d
                    if covers(new, x) is False:
                        out.append(new)
    return out

domains = [df[a].unique().tolist() for a in attrs]
for _, row in df.iterrows():
    x, label = list(row[attrs]), row["EnjoySport"]
    if label == "Yes":
        G = [g for g in G if covers(g, x)]
        S = min_generalizations(S, x)
        G = [g for g in G if all(s is None or s == "?" or g[i] == "?" or g[i] == s for i, s in enumerate(S))]
    else:
        new_G = []
        for g in G:
            if covers(g, x):
                new_G.extend(min_specializations(g, x, domains, S))
            else:
                new_G.append(g)
        G = [g for g in new_G if all(s is None or s == "?" or g[i] == "?" or g[i] == s for i, s in enumerate(S))]
        # Remove duplicates
        G = [list(t) for t in dict.fromkeys(tuple(g) for g in G)]

print("Training data:\n", df)
print("S boundary:", S)
print("G boundary:", G)
plt.bar(attrs, [0 if v is None else 2 if v == "?" else 1 for v in S])
plt.yticks([0, 1, 2], ["Ø", "Specific", "?"])
plt.title("Candidate-Elimination: S Boundary")
plt.grid(axis="y", alpha=.3); plt.show()
