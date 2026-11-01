# Industrial Distribution Network Hosting Capacity – Starter Dataset

Files:
- HESCOM_Industrial_Hosting_Capacity_Synthetic_10000.csv: 10,000 synthetic scenario rows.
- train_gradient_boosting.py: Gradient Boosting regression training script.

Important limitations:
- This is NOT real HESCOM data.
- It is NOT the original dataset from the cited IEEE/IET papers.
- The target Hosting_Capacity_kW is generated using a transparent illustrative screening proxy based on transformer headroom, feeder thermal headroom, voltage headroom, and temperature derating.
- It is suitable for learning the ML workflow and testing code, not for grid planning, equipment selection, or operational decisions.
- For a defensible project result, replace the synthetic rows and proxy labels with authorized HESCOM feeder data and/or OpenDSS load-flow simulations. Determine hosting capacity by incrementally increasing DER penetration until a defined voltage, thermal, protection, or other network constraint is reached.

Run:
1. Install: pip install pandas scikit-learn joblib numpy
2. Keep the CSV and Python script in the same folder.
3. Run: python train_gradient_boosting.py
