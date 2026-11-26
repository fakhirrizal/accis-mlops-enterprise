# ==============================================================================
# tests/test_pipeline.py
# Automated Testing (Quality Gate Kode & API) sebelum Pipeline di-Deploy
# ==============================================================================

from pathlib import Path
import pandas as pd
from fastapi.testclient import TestClient
from data_accis import build_accis_datasets
from lab3_api_serving import app

def test_dataset_schema_and_quality():
    """Memastikan data ACCIS lengkap, tidak kosong, dan nominal selalu positif."""
    build_accis_datasets()
    ref = pd.read_parquet(Path("data/reference_journal.parquet"))
    ang = pd.read_parquet(Path("data/anggaran_biaya.parquet"))

    assert len(ref) > 0, "Data jurnal referensi tidak boleh kosong"
    assert (ref["djo_debit"] > 0).all(), "Nominal debit harus lebih dari 0"
    assert {"tahun", "acc_id", "kategori", "bulan_1", "bulan_12"}.issubset(ang.columns)

def test_api_health_and_validation():
    """Memastikan endpoint /health hidup dan Pydantic menolak nominal negatif."""
    client = TestClient(app)

    resp_health = client.get("/health")
    assert resp_health.status_code == 200
    assert resp_health.json()["status"] == "healthy"

    # Uji kirim nominal minus -> WAJIB ditolak dengan kode HTTP 422 (Unprocessable Entity)
    resp_bad = client.post("/predict-anomaly", json={
        "mjo_code": "JV-TEST-01",
        "mjo_date": "2025-11-26",
        "acc_id": 2,
        "djo_debit": -50000
    })
    assert resp_bad.status_code == 422
