# ==============================================================================
# lab3_api_serving.py
# Real-Time Model Serving & Budget Monitoring REST API dengan FastAPI
# Memuat model @production langsung dari MLflow Registry + Cek Serapan Anggaran
# ==============================================================================

from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE = Path(__file__).parent
mlflow.set_tracking_uri(f"sqlite:///{(BASE / 'mlflow.db').as_posix()}")

# 1. Inisialisasi Aplikasi FastAPI (otomatis membuat dokumentasi Swagger UI di /docs)
app = FastAPI(
    title="ACCIS Enterprise AI & Budget API",
    description="REST API Real-Time untuk deteksi anomali jurnal & cek serapan anggaran (asasta_accis).",
    version="1.0.0"
)

# Muat model terbaik berlabel @production dari MLflow saat server menyala
MODEL_URI = "models:/ACCIS_Anomaly_Model@production"
try:
    model_prod = mlflow.sklearn.load_model(MODEL_URI)
except Exception:
    model_prod = None

# 2. Skema Kontrak Input (Pydantic) -> otomatis menolak data salah tipe/negatif!
class JournalRequest(BaseModel):
    mjo_code: str = Field(..., example="JV-25-99001", description="Nomor bukti jurnal")
    mjo_date: date = Field(..., example="2025-11-26", description="Tanggal transaksi (YYYY-MM-DD)")
    acc_id: int = Field(..., ge=1, le=6, example=3, description="ID akun (1 s/d 6)")
    djo_debit: float = Field(..., gt=0, example=245000000.0, description="Nominal debit dalam Rupiah")

@app.get("/health", tags=["Sistem"])
def health_check():
    """Endpoint pengecekan kesehatan server (wajib ada di standar enterprise)."""
    return {
        "status": "healthy",
        "model_loaded": model_prod is not None,
        "model_uri": MODEL_URI
    }

@app.post("/predict-anomaly", tags=["AI Real-Time Audit"])
def predict_anomaly(payload: JournalRequest):
    """Menilai 1 transaksi jurnal secara real-time sebelum diposting ke buku besar."""
    if model_prod is None:
        raise HTTPException(status_code=503, detail="Model @production belum dilatih. Jalankan lab1_train_mlflow.py dulu.")

    # Susun fitur identik dengan saat training
    X_in = pd.DataFrame([{
        "log_amt": float(np.log1p(payload.djo_debit)),
        "acc_id": payload.acc_id,
        "dow": payload.mjo_date.weekday()
    }])

    skor = float(-model_prod.score_samples(X_in)[0])
    is_anomali = bool(model_prod.predict(X_in)[0] == -1)

    return {
        "mjo_code": payload.mjo_code,
        "djo_debit": payload.djo_debit,
        "anomaly_score": round(skor, 4),
        "is_anomaly": is_anomali,
        "rekomendasi": "HOLD - Perlu verifikasi Kabag Keuangan" if is_anomali else "APPROVE - Pola transaksi wajar"
    }

@app.get("/budget-utilization/{acc_id}/{month}", tags=["Anggaran ACCIS"])
def check_budget_utilization(acc_id: int, month: int):
    """Mengecek realisasi vs pagu tabel anggaran_biaya (bulan_1..bulan_12)."""
    if month < 1 or month > 12:
        raise HTTPException(status_code=400, detail="Bulan harus di antara 1 sampai 12.")

    ang_path = BASE / "data" / "anggaran_biaya.parquet"
    cur_path = BASE / "data" / "current_journal.parquet"
    ref_path = BASE / "data" / "reference_journal.parquet"
    if not ang_path.exists():
        raise HTTPException(status_code=404, detail="Data anggaran belum tersedia.")

    ang = pd.read_parquet(ang_path)
    row = ang[ang["acc_id"] == acc_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Anggaran untuk acc_id={acc_id} tidak ditemukan.")

    pagu_bulan = float(row.iloc[0][f"bulan_{month}"])
    all_jr = pd.concat([pd.read_parquet(ref_path), pd.read_parquet(cur_path)], ignore_index=True)
    realisasi = float(all_jr[(all_jr["acc_id"] == acc_id) & (all_jr["month"] == month)]["djo_debit"].sum())
    persen = round((realisasi / pagu_bulan) * 100, 1) if pagu_bulan > 0 else 0.0

    status = "OVER BUDGET" if persen > 100 else ("WASPADA" if persen >= 85 else "AMAN")
    return {
        "acc_id": acc_id,
        "acc_code": row.iloc[0]["acc_code"],
        "kategori": row.iloc[0]["kategori"],
        "bulan": month,
        "pagu_rp": pagu_bulan,
        "realisasi_rp": realisasi,
        "serapan_persen": persen,
        "status": status
    }