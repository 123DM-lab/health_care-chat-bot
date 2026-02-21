import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/Training.csv")
df = df.dropna(axis=1)

# Encode target
le = LabelEncoder()
df["prognosis"] = le.fit_transform(df["prognosis"])

X = df.drop("prognosis", axis=1)
y = df["prognosis"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model dictionary
models = {
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(),
    "NaiveBayes": GaussianNB()
}

best_model = None
best_accuracy = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name} Accuracy: {acc}")

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

print("Best Model Accuracy:", best_accuracy)

# Save best model
pickle.dump(best_model, open("models/best_model.pkl", "wb"))
pickle.dump(le, open("models/label_encoder.pkl", "wb"))
pickle.dump(X.columns.tolist(), open("models/symptom_list.pkl", "wb"))
