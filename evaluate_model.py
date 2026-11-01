
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("HESCOM_Industrial_Hosting_Capacity_Synthetic_10000.csv")

features = [
    "Transformer_Rating_kVA",
    "Power_Factor",
    "Existing_Peak_Load_kW",
    "Existing_EV_Load_kW",
    "Existing_Solar_PV_kW",
    "Number_of_Nodes",
    "Feeder_Length_km",
    "Total_Resistance_ohm",
    "Average_Impedance_ohm",
    "Minimum_Bus_Voltage_pu",
    "Feeder_Thermal_Limit_kW",
    "Ambient_Temperature_C"
]

X = df[features]
y = df["Hosting_Capacity_kW"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model = joblib.load("hosting_capacity_gradient_boosting.pkl")
predictions = model.predict(X_test)

results = pd.DataFrame({
    "Actual_Hosting_Capacity_kW": y_test.values,
    "Predicted_Hosting_Capacity_kW": predictions
})

results.to_csv("hosting_capacity_predictions.csv", index=False)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, alpha=0.5)
limits = [
    min(y_test.min(), predictions.min()),
    max(y_test.max(), predictions.max())
]
plt.plot(limits, limits, "r--")
plt.xlabel("Actual Hosting Capacity (kW)")
plt.ylabel("Predicted Hosting Capacity (kW)")
plt.title("Actual vs Predicted Hosting Capacity")
plt.grid(True)
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.show()

importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values()

plt.figure(figsize=(9, 6))
importance.plot(kind="barh")
plt.xlabel("Feature Importance")
plt.title("Feature Importance - Gradient Boosting")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

errors = y_test.values - predictions

plt.figure(figsize=(8, 5))
plt.hist(errors, bins=30, edgecolor="black")
plt.xlabel("Prediction Error (kW)")
plt.ylabel("Frequency")
plt.title("Prediction Error Distribution")
plt.tight_layout()
plt.savefig("prediction_error_distribution.png")
plt.show()

print("MAE:", mean_absolute_error(y_test, predictions))
print("RMSE:", np.sqrt(mean_squared_error(y_test, predictions)))
print("R2:", r2_score(y_test, predictions))
print("All graphs and prediction results saved successfully.")
