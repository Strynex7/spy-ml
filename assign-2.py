from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

data = load_digits()
X = data.data
y = data.target

X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pca = PCA(n_components=2)

start = time.time()
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
pca_time = time.time() - start

lda = LinearDiscriminantAnalysis(n_components=2)

start = time.time()
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)
lda_time = time.time() - start

model1 = KNeighborsClassifier(n_neighbors=3)
model1.fit(X_train_pca, y_train)
pca_accuracy = accuracy_score(y_test, model1.predict(X_test_pca))

model2 = KNeighborsClassifier(n_neighbors=3)
model2.fit(X_train_lda, y_train)
lda_accuracy = accuracy_score(y_test, model2.predict(X_test_lda))

print("Original Features:", X.shape[1])
print("Reduced Features:", 2)

print("\nPCA Accuracy:", pca_accuracy * 100, "%")
print("LDA Accuracy:", lda_accuracy * 100, "%")

print("\nPCA Time:", pca_time, "seconds")
print("LDA Time:", lda_time, "seconds")

print("\nAccuracy Difference:",
      abs(lda_accuracy - pca_accuracy) * 100, "%")

print("Time Difference:",
      abs(lda_time - pca_time), "seconds")

print("\nPCA Explained Variance:",
      pca.explained_variance_ratio_.sum() * 100, "%")

if lda_accuracy > pca_accuracy:
    print("LDA gives better classification accuracy")
elif pca_accuracy > lda_accuracy:
    print("PCA gives better classification accuracy")
else:
    print("Both give equal classification accuracy")

if lda_time < pca_time:
    print("LDA is faster in this run")
else:
    print("PCA is faster in this run")
