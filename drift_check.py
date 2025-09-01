import pandas as pd
import mlflow
import evidently
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Load reference & current data
X, y = load_iris(return_X_y=True, as_frame=True)
X_ref, X_cur, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
print(evidently.__version__)
# Create and run drift report
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=X_ref, current_data=X_cur)

# Save report to HTML
report.save("drift_report.html")
with mlflow.start_run():
    mlflow.log_artifact("drift_report.html", artifact_path="drift")

# Extract drift detection result
summary = report.as_dict()
print(summary)

try:
    drift_detected = summary["metrics"][0]["result"]["dataset_drift"]
except KeyError:
    drift_detected = summary["metrics"][0]["result"].get("drift_detected", False)

if drift_detected:
    print("⚠️ Data drift detected!")
    exit(1)
else:
    print("✅ No data drift detected")