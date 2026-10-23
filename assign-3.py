import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

df = pd.read_csv(r"/content/small_taxi_fare_sample.csv")
df.columns = ["distance", "passenger_count", "fare_amount"]

df = df.dropna().drop_duplicates()
df = df[(df["fare_amount"] > 0) & (df["passenger_count"] > 0) & (df["distance"] > 0)]

Q1 = df["fare_amount"].quantile(0.25)
Q3 = df["fare_amount"].quantile(0.75)
IQR = Q3 - Q1

outliers = df[(df["fare_amount"] < Q1 - 1.5 * IQR) |
              (df["fare_amount"] > Q3 + 1.5 * IQR)]

print("Number of Outliers:", len(outliers))

df = df[(df["fare_amount"] >= Q1 - 1.5 * IQR) &
        (df["fare_amount"] <= Q3 + 1.5 * IQR)]

print("\nCorrelation:")
print(df.corr())

X = df[["distance", "passenger_count"]]
y = df["fare_amount"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1)
}

for name, model in models.items():
    start = time.time()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    elapsed = time.time() - start

    print("\n", name)
    print("R2 Score:", r2_score(y_test, pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
    print("MAE:", mean_absolute_error(y_test, pred))
    print("Time:", elapsed, "seconds")
