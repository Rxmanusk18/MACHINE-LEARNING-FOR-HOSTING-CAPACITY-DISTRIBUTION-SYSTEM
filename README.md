# ⚡ Data-Driven Distribution Network Hosting Capacity Prediction

### Machine Learning-Based Hosting Capacity Estimation for EV and Solar PV Integration

## 📌 Project Overview

This project predicts the available hosting capacity of electrical distribution networks using machine learning. It helps estimate how much additional load, such as Electric Vehicle (EV) charging demand and solar photovoltaic (PV) generation, a distribution network can accommodate based on its electrical parameters.

The system uses network characteristics such as transformer rating, existing peak load, power factor, feeder thermal limits, voltage levels, and feeder impedance to estimate hosting capacity.

## 🎯 Objectives

- Predict distribution network hosting capacity in kW.
- Study the impact of electrical network parameters.
- Compare different machine learning regression algorithms.
- Develop an interactive dashboard for hosting capacity prediction.
- Support analysis of EV and renewable energy integration.

## 🧠 Machine Learning Algorithms

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

## 📊 Model Performance

| Algorithm | R² Score | MAE (kW) | RMSE (kW) |
|---|---:|---:|---:|
| Linear Regression | 0.8872 | 21.65 | 30.25 |
| Random Forest | 0.9735 | 8.29 | 14.67 |
| Gradient Boosting | 0.9507 | 14.04 | 19.99 |

*These results are from the synthetic demonstration dataset, not actual HESCOM measurements.*

## 🖥️ Dashboard Features

- Interactive hosting capacity prediction.
- Electrical network parameter inputs.
- Model performance comparison.
- Feature importance visualization.
- Dataset exploration.
- Actual vs. predicted analysis.

## 🛠️ Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Streamlit
- Joblib

## 📁 Project Structure

```text
Hosting_Capacity_Project/
│
├── app.py
├── train_gradient_boosting.py
├── evaluate_model.py
├── compare_models.py
├── HESCOM_Industrial_Hosting_Capacity_Synthetic_10000.csv
├── hosting_capacity_gradient_boosting.pkl
├── hosting_capacity_random_forest.pkl
├── hosting_capacity_predictions.csv
├── model_comparison_results.csv
├── actual_vs_predicted.png
├── feature_importance.png
├── model_comparison.png
├── prediction_error_distribution.png
└── README.md
```

## 🚀 How to Run the Project

1. Clone the repository:

```bash
git clone https://github.com/Rxmanusk18/MACHINE-LEARNING-FOR-HOSTING-CAPACITY-DISTRIBUTION-SYSTEM.git
```

2. Open the project folder:

```bash
cd MACHINE-LEARNING-FOR-HOSTING-CAPACITY-DISTRIBUTION-SYSTEM
```

3. Install the required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib
```

4. Run the dashboard:

```bash
python -m streamlit run app.py
```

## 📈 Future Scope

- Integration of real HESCOM distribution network data.
- Power-flow simulation-based dataset generation.
- Improved prediction using real-time network measurements.
- Analysis of EV charging and solar PV penetration scenarios.
- Deployment for distribution network planning support.

## ⚠️ Dataset Disclaimer

The current project uses a synthetic dataset containing 10,000 records for model development and demonstration. The predictions are not approved grid operating limits. Actual utility data and validated power-flow studies are required before practical engineering application.

## 👨‍💻 Project

**Domain:** Electrical Power Systems | Machine Learning

**Application:** Distribution Network Hosting Capacity Assessment
