import time
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X, y = load_digits(return_X_y=True)
print("Dataset shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def evaluate(Xtr, Xte):
    model = LogisticRegression(max_iter=2000)
    start = time.time()
    model.fit(Xtr, y_train)
    train_time = time.time() - start
    start = time.time()
    pred = model.predict(Xte)
    pred_time = time.time() - start
    return accuracy_score(y_test, pred), train_time, pred_time

base_acc, base_train, base_pred = evaluate(X_train, X_test)
print("Original features:", X_train.shape[1])
print("Accuracy:", round(base_acc, 4))
print("Training time:", round(base_train, 4), "s")

pca_full = PCA().fit(X_train)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)
n95 = int(np.argmax(cum_var >= 0.95) + 1)
print("Components for 95% variance:", n95)

components = sorted(set([2, 5, 10, 20, 30, n95]))
rows = []

for k in components:
    pca = PCA(n_components=k)
    Xtr_p = pca.fit_transform(X_train)
    Xte_p = pca.transform(X_test)
    acc, tr_time, pr_time = evaluate(Xtr_p, Xte_p)
    rows.append({
        "Components": k,
        "Variance Retained": round(pca.explained_variance_ratio_.sum(), 4),
        "Accuracy": round(acc, 4),
        "Train Time (s)": round(tr_time, 4),
        "Predict Time (s)": round(pr_time, 5)
    })

rows.append({
    "Components": X_train.shape[1],
    "Variance Retained": 1.0,
    "Accuracy": round(base_acc, 4),
    "Train Time (s)": round(base_train, 4),
    "Predict Time (s)": round(base_pred, 5)
})

results = pd.DataFrame(rows)
print(results)

# Feature Transformation- 
# Apply PCA algorithm for dimensionality reduction on suitable open access high 
# dimensional dataset and observe the influence of feature reduction on model 
# performance and computational efficiency.  
