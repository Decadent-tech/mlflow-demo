# drift_check.py
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Load reference & current data
X, y = load_iris(return_X_y=True, as_frame=True)
X_ref, X_cur, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# Drift report
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=X_ref, current_data=X_cur)
# Save HTML report
report.save_html("drift_report.html")
summary = report.as_dict()
print(summary)

if summary["metrics"][0]["result"]["dataset_drift"]:
    print("⚠️ Data drift detected!")
    exit(1)  # Fail workflow if drift found
else:
    print("✅ No data drift detected")
