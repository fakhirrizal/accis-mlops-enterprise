# ==============================================================================
# lab2_monitor_drift.py
# Deteksi Data Drift antara Data Referensi (Jan-Sep) vs Data Baru (Okt-Des)
# Menggunakan Evidently AI (Laporan HTML Interaktif + Keputusan Retraining)
# ==============================================================================

from pathlib import Path
import json
import pandas as pd

# Dukungan kompatibilitas lintas versi Evidently (v0.4–v0.6 dan v0.7+)
try:
    from evidently import Report
    from evidently.presets import DataDriftPreset
    EVIDENTLY_V7 = True
except ImportError:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    EVIDENTLY_V7 = False

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
REPORT_DIR = BASE / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def run_drift_check():
    # Muat data referensi (saat model dilatih) & data baru (produksi saat ini)
    cols = ["djo_debit", "log_amt", "acc_id", "dow", "of_id"]
    ref = pd.read_parquet(DATA_DIR / "reference_journal.parquet")[cols]
    cur = pd.read_parquet(DATA_DIR / "current_journal.parquet")[cols]

    # Jalankan evaluasi Data Drift (KS-Test / Wasserstein / PSI otomatis per kolom)
    report = Report(metrics=[DataDriftPreset()])

    if EVIDENTLY_V7:
        eval_res = report.run(current_data=cur, reference_data=ref)
        html_path = REPORT_DIR / "accis_drift_report.html"
        eval_res.save_html(str(html_path))
        result_dict = json.loads(eval_res.json())
        # Cek ringkasan drift di format v0.7+
        drift_summary = result_dict
    else:
        report.run(reference_data=ref, current_data=cur)
        html_path = REPORT_DIR / "accis_drift_report.html"
        report.save_html(str(html_path))
        result_dict = report.as_dict()
        drift_summary = result_dict["metrics"][0]["result"]

    # Simpan ringkasan JSON agar bisa dibaca otomatis oleh CI/CD (GitHub Actions)
    json_path = REPORT_DIR / "drift_summary.json"
    json_path.write_text(json.dumps(drift_summary, indent=2, default=str), encoding="utf-8")

    print(f"Laporan visual Data Drift tersimpan di: {html_path}")
    print(f"Rata-rata nominal Referensi : Rp {ref['djo_debit'].mean():,.0f}")
    print(f"Rata-rata nominal Saat Ini  : Rp {cur['djo_debit'].mean():,.0f}")
    selisih_pct = ((cur["djo_debit"].mean() - ref["djo_debit"].mean()) / ref["djo_debit"].mean()) * 100
    print(f"Perubahan rata-rata nominal : {selisih_pct:+.1f}%")

    if abs(selisih_pct) >= 25:
        print("STATUS: DRIFT TERDETEKSI! Direkomendasikan menjalankan retraining model di MLflow.")
    else:
        print("STATUS: Distribusi data stabil. Model @production masih layak dipakai.")

if __name__ == "__main__":
    run_drift_check()