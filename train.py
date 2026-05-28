import pandas as pd

import numpy as np

import joblib

import json
 
from sklearn.tree import DecisionTreeClassifier, export_text

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (

    accuracy_score, classification_report,

    confusion_matrix, roc_auc_score

)
 
# ── 1. Load data ──────────────────────────────────────────────────────────────

df = pd.read_csv("employee_salary_data.csv")
 
# ── 2. Feature engineering ────────────────────────────────────────────────────

edu_order = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}

df["Education_Enc"] = df["Education"].map(edu_order)
 
dept_enc = LabelEncoder()

df["Department_Enc"] = dept_enc.fit_transform(df["Department"])
 
features = ["Age", "Education_Enc", "Experience_Years", "Num_Skills", "Department_Enc"]

target   = "Salary_Label"
 
X = df[features]

y = df[target]
 
# ── 3. Train / Test split ─────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.2, random_state=42, stratify=y

)
 
# ── 4. Train Decision Tree ────────────────────────────────────────────────────

model = DecisionTreeClassifier(

    max_depth=8,

    min_samples_split=20,

    min_samples_leaf=10,

    criterion="gini",

    random_state=42

)

model.fit(X_train, y_train)
 
# ── 5. Evaluate ───────────────────────────────────────────────────────────────

y_pred = model.predict(X_test)

acc    = accuracy_score(y_test, y_pred)

cv     = cross_val_score(model, X, y, cv=5, scoring="accuracy")
 
print(f"\n{'='*50}")

print(f"  Decision Tree — Employee Salary Prediction")

print(f"{'='*50}")

print(f"  Test Accuracy  : {acc*100:.2f}%")

print(f"  CV Accuracy    : {cv.mean()*100:.2f}% ± {cv.std()*100:.2f}%")

print(f"\n{classification_report(y_test, y_pred)}")

print("Confusion Matrix:")

print(confusion_matrix(y_test, y_pred))
 
# Feature importance

fi = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)

print(f"\nFeature Importances:\n{fi.to_string()}")
 
# ── 6. Save artifacts ─────────────────────────────────────────────────────────

joblib.dump(model, "salary_model.pkl")

joblib.dump(dept_enc, "dept_encoder.pkl")
 
meta = {

    "features": features,

    "accuracy": round(acc * 100, 2),

    "cv_accuracy": round(cv.mean() * 100, 2),

    "cv_std": round(cv.std() * 100, 2),

    "departments": list(dept_enc.classes_),

    "feature_importances": dict(zip(features, model.feature_importances_.tolist())),

    "classes": list(model.classes_),

    "tree_depth": int(model.get_depth()),

    "n_leaves": int(model.get_n_leaves()),

}

with open("model_meta.json", "w") as f:

    json.dump(meta, f, indent=2)
 
print(f"\nArtifacts saved: salary_model.pkl, dept_encoder.pkl, model_meta.json")
 