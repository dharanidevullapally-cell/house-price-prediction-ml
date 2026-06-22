import pandas as pd
import joblib
import matplotlib.pyplot as plt

model = joblib.load("models/house_model.pkl")
features = joblib.load("models/features.pkl")

importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(10,6))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance Analysis")

plt.tight_layout()

plt.show()