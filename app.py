
import os
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="GridSense | Hosting Capacity Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_FILE = "HESCOM_Industrial_Hosting_Capacity_Synthetic_10000.csv"
RF_FILE = "hosting_capacity_random_forest.pkl"
GB_FILE = "hosting_capacity_gradient_boosting.pkl"
RESULT_FILE = "model_comparison_results.csv"

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

# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #07111f 0%, #0b1930 55%, #101d35 100%);
    color: #e8f0ff;
}

[data-testid="stSidebar"] {
    background: #081426;
    border-right: 1px solid #1c3558;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

h1, h2, h3 {
    color: #edf5ff !important;
}

p, label, span {
    color: #c6d5eb;
}

.hero {
    background: linear-gradient(110deg, #102c50, #164b75, #087c91);
    border: 1px solid #28618a;
    border-radius: 18px;
    padding: 28px;
    margin-bottom: 24px;
}

.hero h1 {
    font-size: 34px;
    margin-bottom: 8px;
    color: white !important;
}

.hero p {
    color: #d5eaff;
    font-size: 15px;
}

.metric-card {
    background: linear-gradient(145deg, #142944, #102038);
    border: 1px solid #284666;
    border-radius: 15px;
    padding: 20px;
    min-height: 125px;
}

.metric-label {
    color: #9bb7d8;
    font-size: 13px;
    margin-bottom: 10px;
}

.metric-value {
    color: #ffffff;
    font-size: 27px;
    font-weight: 700;
}

.metric-caption {
    color: #8ed8e8;
    font-size: 12px;
    margin-top: 6px;
}

.section-card {
    background: #10213a;
    border: 1px solid #243f60;
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 15px;
}

div[data-testid="stMetric"] {
    background: #122742;
    border: 1px solid #294563;
    padding: 15px;
    border-radius: 13px;
}

div[data-testid="stMetricValue"] {
    color: #ffffff;
}

.stButton button {
    background: linear-gradient(90deg, #087cba, #08a6b5);
    color: white;
    border: 0;
    border-radius: 10px;
    font-weight: 700;
    min-height: 45px;
}

.stButton button:hover {
    border: 1px solid #65d8ef;
    color: white;
}

div[data-testid="stForm"] {
    background: #10213a;
    padding: 20px;
    border: 1px solid #294563;
    border-radius: 15px;
}

hr {
    border-color: #294563;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# ---------------- DATA ----------------

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_FILE)

    if not all(c in data.columns for c in FEATURES + [TARGET]):
        raise ValueError("Dataset columns do not match the model.")

    if data[FEATURES + [TARGET]].isnull().any().any():
        raise ValueError("Dataset contains missing values.")

    return data


# ---------------- MODEL TRAINING ----------------

@st.cache_resource
def load_models(data_file):

    data = pd.read_csv(data_file)

    X = data[FEATURES]
    y = data[TARGET]

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

        results.append({
            "Model": name,
            "R2 Score": r2_score(y_test, predictions),
            "MAE (kW)": mean_absolute_error(y_test, predictions),
            "RMSE (kW)": np.sqrt(
                mean_squared_error(y_test, predictions)
            ),
            "Training Time (seconds)": training_time
        })

    joblib.dump(models["Random Forest"], RF_FILE)
    joblib.dump(models["Gradient Boosting"], GB_FILE)

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULT_FILE, index=False)

    return models, results_df, X_test, y_test


# ---------------- LOAD PROJECT ----------------

if not os.path.exists(DATA_FILE):
    st.error("Dataset file not found. Keep the CSV in the project folder.")
    st.stop()

try:
    df = load_data()
    models, results_df, X_test, y_test = load_models(DATA_FILE)
except Exception as e:
    st.error(f"Unable to load project: {e}")
    st.stop()

rf_model = models["Random Forest"]
gb_model = models["Gradient Boosting"]

rf_predictions = rf_model.predict(X_test)

rf_r2 = r2_score(y_test, rf_predictions)
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))

# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.markdown("## ⚡ GRIDSENSE")
    st.caption("POWER SYSTEM ANALYTICS")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Hosting Capacity Prediction",
            "Model Comparison",
            "Network Analytics",
            "Dataset Explorer",
            "About Project"
        ]
    )

    st.divider()

    st.markdown("### MODEL STATUS")
    st.success("Random Forest: Ready")
    st.success("Gradient Boosting: Ready")

    st.caption("Dataset: 10,000 synthetic scenarios")
    st.caption("Application: Industrial Distribution Network")

# ---------------- HERO ----------------

st.markdown("""
<div class="hero">
    <h1>⚡ GridSense Analytics</h1>
    <p>Data-Driven Distribution Network Hosting Capacity Prediction</p>
    <p>Intelligent electrical network analysis powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ---------------- OVERVIEW ----------------

if page == "Overview":

    st.subheader("Network Intelligence Overview")

    st.write(
        "Analyze electrical network parameters, estimate hosting capacity "
        "and compare machine learning algorithms."
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Dataset Records", f"{len(df):,}")

    with c2:
        st.metric("Best Model R²", f"{rf_r2:.4f}")

    with c3:
        st.metric("Mean Absolute Error", f"{rf_mae:.2f} kW")

    with c4:
        st.metric("RMSE", f"{rf_rmse:.2f} kW")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.markdown("### Hosting Capacity Distribution")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df[TARGET], bins=30)
        ax.set_xlabel("Hosting Capacity (kW)")
        ax.set_ylabel("Number of Scenarios")
        ax.set_title("Distribution of Hosting Capacity")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.markdown("### Model Performance")

        chart_data = results_df.set_index("Model")[["R2 Score"]]
        st.bar_chart(chart_data, y="R2 Score")

    st.divider()

    st.markdown("### Project Highlights")

    a, b, c = st.columns(3)

    with a:
        st.markdown("#### ⚡ Network Analysis")
        st.write("Uses transformer, feeder, voltage and load parameters.")

    with b:
        st.markdown("#### 🤖 Machine Learning")
        st.write("Compares three regression algorithms.")

    with c:
        st.markdown("#### 📈 Prediction")
        st.write("Estimates additional hosting capacity for a given scenario.")

# ---------------- PREDICTION ----------------

elif page == "Hosting Capacity Prediction":

    st.subheader("Hosting Capacity Prediction")

    st.write(
        "Enter the electrical parameters of an industrial distribution "
        "network to estimate its hosting capacity."
    )

    defaults = df[FEATURES].median()

    with st.form("network_prediction_form"):

        st.markdown("### Transformer and Load Parameters")

        c1, c2, c3 = st.columns(3)

        with c1:
            transformer = st.number_input(
                "Transformer Rating (kVA)",
                min_value=1.0,
                value=float(defaults["Transformer_Rating_kVA"])
            )

            pf = st.number_input(
                "Power Factor",
                min_value=0.1,
                max_value=1.0,
                value=float(defaults["Power_Factor"])
            )

            peak = st.number_input(
                "Existing Peak Load (kW)",
                min_value=0.0,
                value=float(defaults["Existing_Peak_Load_kW"])
            )

            ev = st.number_input(
                "Existing EV Load (kW)",
                min_value=0.0,
                value=float(defaults["Existing_EV_Load_kW"])
            )

        with c2:
            solar = st.number_input(
                "Existing Solar PV (kW)",
                min_value=0.0,
                value=float(defaults["Existing_Solar_PV_kW"])
            )

            nodes = st.number_input(
                "Number of Nodes",
                min_value=1,
                value=int(defaults["Number_of_Nodes"])
            )

            length = st.number_input(
                "Feeder Length (km)",
                min_value=0.01,
                value=float(defaults["Feeder_Length_km"])
            )

            resistance = st.number_input(
                "Total Resistance (ohm)",
                min_value=0.0,
                value=float(defaults["Total_Resistance_ohm"])
            )

        with c3:
            impedance = st.number_input(
                "Average Impedance (ohm)",
                min_value=0.0,
                value=float(defaults["Average_Impedance_ohm"])
            )

            voltage = st.number_input(
                "Minimum Bus Voltage (pu)",
                min_value=0.1,
                max_value=1.2,
                value=float(defaults["Minimum_Bus_Voltage_pu"])
            )

            thermal = st.number_input(
                "Feeder Thermal Limit (kW)",
                min_value=1.0,
                value=float(defaults["Feeder_Thermal_Limit_kW"])
            )

            temperature = st.number_input(
                "Ambient Temperature (°C)",
                value=float(defaults["Ambient_Temperature_C"])
            )

        submitted = st.form_submit_button(
            "⚡ PREDICT HOSTING CAPACITY",
            use_container_width=True
        )

    if submitted:

        input_data = {
            "Transformer_Rating_kVA": transformer,
            "Power_Factor": pf,
            "Existing_Peak_Load_kW": peak,
            "Existing_EV_Load_kW": ev,
            "Existing_Solar_PV_kW": solar,
            "Number_of_Nodes": nodes,
            "Feeder_Length_km": length,
            "Total_Resistance_ohm": resistance,
            "Average_Impedance_ohm": impedance,
            "Minimum_Bus_Voltage_pu": voltage,
            "Feeder_Thermal_Limit_kW": thermal,
            "Ambient_Temperature_C": temperature
        }

        input_df = pd.DataFrame([input_data], columns=FEATURES)

        rf_result = float(rf_model.predict(input_df)[0])
        gb_result = float(gb_model.predict(input_df)[0])

        st.divider()
        st.markdown("### Prediction Results")

        if rf_result < 0:
            st.warning(
                f"Random Forest returned {rf_result:.2f} kW. "
                "A negative capacity is not physically meaningful."
            )
        else:
            st.success("Prediction completed.")

        p1, p2 = st.columns(2)

        with p1:
            st.metric(
                "Random Forest Estimate",
                f"{max(0, rf_result):.2f} kW"
            )

        with p2:
            st.metric(
                "Gradient Boosting Estimate",
                f"{max(0, gb_result):.2f} kW"
            )

        st.markdown("### Model Prediction Comparison")

        prediction_comparison = pd.DataFrame({
            "Model": ["Random Forest", "Gradient Boosting"],
            "Estimated Capacity (kW)": [rf_result, gb_result]
        }).set_index("Model")

        st.bar_chart(prediction_comparison)

        with st.expander("View network input parameters"):
            st.dataframe(input_df, use_container_width=True)

        st.warning(
            "This estimate is based on a synthetic training dataset. "
            "It is not an actual HESCOM network limit or an approved "
            "connection capacity. Validate with real network data and "
            "power-flow studies before practical use."
        )

# ---------------- MODEL COMPARISON ----------------

elif page == "Model Comparison":

    st.subheader("Machine Learning Model Comparison")

    st.write(
        "Performance of three regression algorithms using the same "
        "80:20 train-test split."
    )

    best = results_df.loc[results_df["R2 Score"].idxmax()]

    st.success(f"Best-performing model: {best['Model']}")

    c1, c2, c3 = st.columns(3)

    for col, (_, row) in zip(
        [c1, c2, c3],
        results_df.iterrows()
    ):
        with col:
            st.metric(
                row["Model"],
                f"{row['R2 Score']:.4f} R²",
                f"MAE: {row['MAE (kW)']:.2f} kW"
            )

    st.divider()

    st.markdown("### R² Score Comparison")

    st.bar_chart(
        results_df.set_index("Model")[["R2 Score"]]
    )

    st.markdown("### Prediction Error Comparison")

    st.bar_chart(
        results_df.set_index("Model")[["MAE (kW)", "RMSE (kW)"]]
    )

    st.markdown("### Training Time Comparison")

    st.bar_chart(
        results_df.set_index("Model")[["Training Time (seconds)"]]
    )

    st.markdown("### Detailed Evaluation Results")

    st.dataframe(
        results_df.round(4),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "R² measures goodness of fit. MAE and RMSE measure prediction "
        "error in kW. Training times depend on the computer used."
    )

# ---------------- NETWORK ANALYTICS ----------------

elif page == "Network Analytics":

    st.subheader("Electrical Network Analytics")

    st.write(
        "Explore the relationship between network parameters "
        "and the synthetic hosting-capacity target."
    )

    selected_feature = st.selectbox(
        "Select electrical parameter",
        FEATURES
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.scatter(
        df[selected_feature],
        df[TARGET],
        alpha=0.35
    )

    ax.set_xlabel(selected_feature)
    ax.set_ylabel("Hosting Capacity (kW)")
    ax.set_title("Network Parameter vs Hosting Capacity")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    st.markdown("### Random Forest Feature Importance")

    importance = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": rf_model.feature_importances_
    }).sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.barh(
        importance["Feature"],
        importance["Importance"]
    )

    ax.set_xlabel("Importance")
    ax.set_title("Random Forest Feature Importance")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(
        importance.sort_values(
            "Importance", ascending=False
        ).round(4),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Feature importance indicates how much the trained model "
        "relies on each feature. It does not prove physical causation."
    )

    st.divider()

    st.markdown("### Actual vs Predicted Values")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(y_test, rf_predictions, alpha=0.5)

    low = min(float(y_test.min()), float(rf_predictions.min()))
    high = max(float(y_test.max()), float(rf_predictions.max()))

    ax.plot([low, high], [low, high], linestyle="--")

    ax.set_xlabel("Actual Capacity (kW)")
    ax.set_ylabel("Predicted Capacity (kW)")
    ax.set_title("Random Forest: Actual vs Predicted")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ---------------- DATASET EXPLORER ----------------

elif page == "Dataset Explorer":

    st.subheader("Dataset Explorer")

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Total Features", len(FEATURES))
    c3.metric("Target Variable", "Hosting Capacity")

    st.markdown("### Dataset Preview")

    st.dataframe(
        df.head(100),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Statistical Summary")

    st.dataframe(
        df[FEATURES + [TARGET]].describe().T,
        use_container_width=True
    )

    st.markdown("### Download Dataset")

    st.download_button(
        "Download CSV Dataset",
        data=df.to_csv(index=False),
        file_name="hosting_capacity_dataset.csv",
        mime="text/csv"
    )

# ---------------- ABOUT ----------------

elif page == "About Project":

    st.subheader("About the Project")

    st.markdown("""
    ### Data-Driven Distribution Network Hosting Capacity Prediction

    **Application:** Industrial Distribution Networks

    **Machine Learning:** Linear Regression, Random Forest,
    Gradient Boosting

    **Primary Model:** Random Forest Regressor

    **Dataset:** 10,000 synthetic network scenarios

    ### Electrical Parameters

    - Transformer rating
    - Power factor
    - Existing peak load
    - Existing EV load
    - Solar PV capacity
    - Number of network nodes
    - Feeder length
    - Total resistance
    - Average impedance
    - Minimum bus voltage
    - Feeder thermal limit
    - Ambient temperature

    ### Project Goal

    Estimate additional hosting capacity from distribution network
    parameters using machine learning.
    """)

    st.warning(
        "The current dataset is synthetic and is not actual HESCOM data. "
        "The generated target is an illustrative screening proxy, not "
        "a validated power-flow result. Real utility data and network "
        "simulation are required before operational decisions."
    )

# ---------------- FOOTER ----------------

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#91a9c7; padding:15px;">
        ⚡ <b>GRIDSENSE ANALYTICS</b><br>
        Distribution Network Hosting Capacity Prediction<br>
        <small>Machine Learning Research Prototype</small>
    </div>
    """,
    unsafe_allow_html=True
)
