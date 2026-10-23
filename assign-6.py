import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

iris = load_iris()
X = iris.data
print("Dataset: Iris")
print("Dataset shape:", X.shape)
print("Features:", iris.feature_names)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k_values = list(range(1, 11))
inertia = []
for k in k_values:
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    model.fit(X_scaled)
    inertia.append(model.inertia_)

print()
print("ELBOW METHOD")
print("K   Inertia (WCSS)")
for k, value in zip(k_values, inertia):
    print(k, "  ", round(value, 2))

ks = np.array(k_values)
wcss = np.array(inertia)
x1, y1 = ks[0], wcss[0]
x2, y2 = ks[-1], wcss[-1]
distance = abs((y2 - y1) * ks - (x2 - x1) * wcss + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
best_k = int(ks[np.argmax(distance)])
print()
print("Optimal number of clusters (elbow point):", best_k)

final_model = KMeans(n_clusters=best_k, n_init=10, random_state=42)
labels = final_model.fit_predict(X_scaled)

print()
print("FINAL K-MEANS RESULT")
print("Inertia:", round(final_model.inertia_, 2))
print("Silhouette score:", round(silhouette_score(X_scaled, labels), 4))

print()
print("Number of points in each cluster:")
print(pd.Series(labels).value_counts().sort_index())

print()
print("Cluster centers (original units):")
centers = scaler.inverse_transform(final_model.cluster_centers_)
print(pd.DataFrame(centers, columns=iris.feature_names).round(2))

print()
print("Clusters vs actual species:")
species = pd.Series(iris.target_names[iris.target], name="Species")
print(pd.crosstab(pd.Series(labels, name="Cluster"), species))
