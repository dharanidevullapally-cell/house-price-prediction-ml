import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# Load Dataset
df = pd.read_csv("data/Housing.csv")

# Convert categorical columns
df = pd.get_dummies(
    df,
    drop_first=True
)

# Features
X = df.drop("price", axis=1)

# Target
y = df["price"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Random Forest
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

# Metrics
r2 = r2_score(y_test, predictions)

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

print(f"R2 Score : {r2:.4f}")
print(f"MAE      : {mae:.2f}")
print(f"RMSE     : {rmse:.2f}")

# Feature Importance
print("\nFeature Importance:\n")

importance = model.feature_importances_

for feature, score in sorted(
    zip(X.columns, importance),
    key=lambda x: x[1],
    reverse=True
):
    print(
        f"{feature:<30} {score:.4f}"
    )

# Save model
joblib.dump(
    model,
    "models/house_model.pkl"
)

# Save feature names too
joblib.dump(
    X.columns.tolist(),
    "models/features.pkl"
)

print("\nModel Saved Successfully!")