import time
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

data = load_breast_cancer()
X = data.data
y = data.target
names = list(data.target_names)

print("Dataset: Breast Cancer Wisconsin")
print("Dataset shape:", X.shape)
print("Class counts:")
print(pd.Series(data.target_names[y]).value_counts())

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

models = {
    "Decision Tree (baseline)": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

rows = []
rf_pred = None
for name, model in models.items():
    start = time.time()
    model.fit(x_train, y_train)
    train_time = time.time() - start
    pred = model.predict(x_test)
    cv_score = cross_val_score(model, x_train, y_train, cv=5).mean()
    rows.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, pred), 4),
        "Precision": round(precision_score(y_test, pred), 4),
        "Recall": round(recall_score(y_test, pred), 4),
        "F1 Score": round(f1_score(y_test, pred), 4),
        "CV Accuracy": round(cv_score, 4),
        "Train Time (s)": round(train_time, 4),
    })
    if name == "Random Forest":
        rf_model = model
        rf_pred = pred

results = pd.DataFrame(rows)
print()
print(results.to_string(index=False))

base_acc = results.loc[0, "Accuracy"]
rf_acc = results.loc[2, "Accuracy"]
print()
print("Random Forest vs baseline accuracy:", rf_acc, "vs", base_acc)
print("Improvement over baseline:", round((rf_acc - base_acc) * 100, 2), "percentage points")
print("Best model overall:", results.loc[results["Accuracy"].idxmax(), "Model"])

print()
print("Random Forest Confusion Matrix:")
cm = confusion_matrix(y_test, rf_pred)
print(pd.DataFrame(cm, index=["Actual " + n for n in names], columns=["Pred " + n for n in names]))

print()
print("Random Forest Classification Report:")
print(classification_report(y_test, rf_pred, target_names=names))

print("Top 5 important features:")
importance = pd.Series(rf_model.feature_importances_, index=data.feature_names)
print(importance.sort_values(ascending=False).head(5).round(4))



 
# Ensemble Learning- 
# Make use of Random Forest Classifier model on suitable open access dataset for 
# monitoring the performance in comparison with a baseline machine learning model.  
