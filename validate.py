# validate.py
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import sys

# Load data
X, y = load_iris(return_X_y=True)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Get last run
client = mlflow.tracking.MlflowClient(tracking_uri="file:./mlruns")
experiment = client.get_experiment_by_name("iris_experiment")
runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time desc"], max_results=1)

if not runs:
    print("No runs found!")
    sys.exit(1)

run_id = runs[0].info.run_id
model_uri = f"runs:/{run_id}/model"

# Load model & evaluate
model = mlflow.sklearn.load_model(model_uri)
preds = model.predict(X_test)

acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds, average="weighted")

print(f"Validation: Accuracy={acc:.4f}, F1={f1:.4f}")

# Threshold check
if acc < 0.85 or f1 < 0.85:
    print("❌ Model validation failed!")
    sys.exit(1)

print("✅ Model passed validation")
