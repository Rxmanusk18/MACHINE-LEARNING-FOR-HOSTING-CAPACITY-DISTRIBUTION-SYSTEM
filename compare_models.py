
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_FILE = "HESCOM_Industrial_Hosting_Capacity_Synthetic_10000.csv"

df = pd.read_csv(DATA_FILE)

FEATURES = [
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

TARGET = "Hosting_Capacity_kW"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        loss="squared_error",
        random_state=42
    )
}

results = []

for name, model in models.items():

    start = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "R2 Score": r2,
        "MAE (kW)": mae,
        "RMSE (kW)": rmse,
        "Training Time (seconds)": training_time
    })

    print(f"\n{name}")
    print(f"R2 Score: {r2:.4f}")
    print(f"MAE: {mae:.3f} kW")
    print(f"RMSE: {rmse:.3f} kW")
    print(f"Training Time: {training_time:.3f} seconds")

results_df = pd.DataFrame(results)

results_df.to_csv("model_comparison_results.csv", index=False)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

best_model = results_df.loc[results_df["R2 Score"].idxmax()]

print("\nBest model based on R2 Score:")
print(best_model["Model"])

fig, ax = plt.subplots(figsize=(9, 5))

ax.bar(results_df["Model"], results_df["R2 Score"])

ax.set_title("Comparison of Machine Learning Models")
ax.set_ylabel("R2 Score")
ax.set_xlabel("Machine Learning Algorithm")
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=300)
plt.show()
