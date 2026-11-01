"""
Gradient Boosting baseline for the accompanying synthetic hosting-capacity dataset.

IMPORTANT:
This dataset is synthetic and uses a transparent screening-proxy label, not
HESCOM measurements or OpenDSS power-flow results. Replace it with validated
utility/simulation data before making engineering or planning decisions.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import numpy as np

DATA_FILE = "HESCOM_Industrial_Hosting_Capacity_Synthetic_10000.csv"

df = pd.read_csv(DATA_FILE)

target = "Hosting_Capacity_kW"
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
    "Ambient_Temperature_C",
]

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    loss="squared_error",
    random_state=42
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print(f"Rows: {len(df)}")
print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")
print(f"MAE: {mae:.3f} kW")
print(f"RMSE: {rmse:.3f} kW")
print(f"R2: {r2:.4f}")

joblib.dump(model, "hosting_capacity_gradient_boosting.pkl")
print("Saved model: hosting_capacity_gradient_boosting.pkl")
