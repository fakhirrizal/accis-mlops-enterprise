# ==============================================================================
# lab1_train_mlflow.py
# Eksperimen Sistematis & Model Registry menggunakan MLflow (Kompatibel MLflow 3.x)
# ==============================================================================

from pathlib import Path
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"

# 1. Arahkan MLflow ke database lokal SQLite agar mendukung Model Registry
mlflow.set_tracking_uri(f"sqlite:///{(BASE / 'mlflow.db').as_posix()}")
mlflow.set_experiment("ACCIS_Audit_Anomaly_Detection")

# 2. Muat data referensi (Jan-Sep 2025)
df = pd.read_parquet(DATA_DIR / "reference_journal.parquet")
features = ["log_amt", "acc_id", "dow"]
X = df[features]
y_true = df["is_anomaly"]

# 3. Uji 3 kandidat hyperparameter
kandidat_eksperimen = [
    {"n_estimators": 100, "contamination": 0.010, "max_samples": 256},
    {"n_estimators": 200, "contamination": 0.015, "max_samples": 256},
    {"n_estimators": 300, "contamination": 0.020, "max_samples": "auto"},
]

best_f1 = -1.0
best_model_uri = None

for idx, params in enumerate(kandidat_eksperimen, start=1):
    run_name = f"IsoForest_Run_{idx}_contam_{params['contamination']}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_param("features", ",".join(features))
        mlflow.log_param("n_rows_train", len(df))

        model = IsolationForest(
            n_estimators=params["n_estimators"],
            contamination=params["contamination"],
            max_samples=params["max_samples"],
            random_state=42
        ).fit(X)

        y_pred = (model.predict(X) == -1).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)

        mlflow.log_metrics({"precision": prec, "recall": rec, "f1_score": f1})

        # Pakai serialization_format="cloudpickle" agar tidak ditolak skops di MLflow 3.x
        try:
            model_info = mlflow.sklearn.log_model(
                model, name="model", serialization_format="cloudpickle"
            )
        except TypeError:
            model_info = mlflow.sklearn.log_model(
                model, artifact_path="model", serialization_format="cloudpickle"
            )

        print(f"[Run #{idx}] {run_name} -> Precision={prec:.3f} | Recall={rec:.3f} | F1={f1:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model_uri = model_info.model_uri

# 4. Daftarkan model terbaik ke MLflow Model Registry pakai URI dari model_info
registered = mlflow.register_model(model_uri=best_model_uri, name="ACCIS_Anomaly_Model")

client = mlflow.tracking.MlflowClient()
client.set_registered_model_alias(
    name="ACCIS_Anomaly_Model",
    alias="production",
    version=registered.version
)

print(f"\nModel terbaik (F1={best_f1:.3f}) didaftarkan sebagai 'ACCIS_Anomaly_Model' v{registered.version} [@production]")