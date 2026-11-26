# ==============================================================================
# data_accis.py
# Generator Data Sintetis Sesuai Skema asasta_accis.sql + Simulasi Data Drift
# Tabel: masteraccount_perkiraan, mjournal, djournal, anggaran_biaya
# ==============================================================================

from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def build_accis_datasets(seed: int = 42):
    rng = np.random.default_rng(seed)

    # 1. Master Akun (meniru tabel masteraccount_perkiraan)
    acc = pd.DataFrame({
        "acc_id": [1, 2, 3, 4, 5, 6],
        "acc_code": ["5101", "5102", "5201", "5202", "5301", "4101"],
        "acc_name": [
            "Beban Gaji", "Beban Listrik & Energi", "Beban Pemeliharaan Pipa",
            "Beban Bahan Kimia Air", "Beban Administrasi Umum", "Pendapatan Air"
        ],
        "kategori_default": ["GAJI", "KEGIATAN", "KEGIATAN", "BARANG", "BARANG", "KEGIATAN"]
    })

    # 2. Tabel Anggaran Biaya (meniru tabel anggaran_biaya di asasta_accis.sql)
    # Menyimpan pagu per tahun, bagian_id, acc_id, kategori, dan bulan_1..bulan_12
    rows_ang = []
    for _, r in acc[acc["acc_code"].str.startswith("5")].iterrows():
        base_bln = rng.uniform(80_000_000, 250_000_000)
        bulanan = {f"bulan_{m}": round(base_bln * rng.uniform(0.9, 1.15), 0) for m in range(1, 13)}
        rows_ang.append({
            "tahun": 2025,
            "bagian_id": int(rng.integers(1, 5)),
            "acc_id": int(r["acc_id"]),
            "acc_code": r["acc_code"],
            "kategori": r["kategori_default"],
            "total": sum(bulanan.values()),
            **bulanan
        })
    anggaran_df = pd.DataFrame(rows_ang)

    # 3. Data Referensi (Periode Normal: Jan - Sep 2025) -> dipakai melatih model
    n_ref = 2400
    tgl_ref = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 270, n_ref), unit="D")
    ref_df = pd.DataFrame({
        "djo_id": np.arange(1, n_ref + 1),
        "mjo_code": [f"JV-25-{i:05d}" for i in range(1, n_ref + 1)],
        "mjo_date": tgl_ref,
        "acc_id": rng.integers(1, 7, n_ref),
        "of_id": rng.integers(1, 5, n_ref),
        "djo_debit": np.round(rng.lognormal(14.5, 0.55, n_ref), 0),
        "input_by": rng.choice(["siti", "budi", "andi", "rina"], n_ref)
    })
    # Sisipkan 1.5% anomali bisnis nyata pada data referensi
    bad_ref = rng.choice(n_ref, int(n_ref * 0.015), replace=False)
    ref_df.loc[bad_ref, "djo_debit"] *= rng.uniform(40, 75, len(bad_ref))
    ref_df["is_anomaly"] = 0
    ref_df.loc[bad_ref, "is_anomaly"] = 1

    # 4. Data Produksi Baru (Periode Tutup Buku: Okt - Des 2025) -> SENGAJA ADA DATA DRIFT!
    # Simulasi lonjakan belanja akhir tahun (+65% nominal) & banyak input di akhir pekan
    n_cur = 800
    tgl_cur = pd.Timestamp("2025-10-01") + pd.to_timedelta(rng.integers(0, 90, n_cur), unit="D")
    cur_df = pd.DataFrame({
        "djo_id": np.arange(n_ref + 1, n_ref + n_cur + 1),
        "mjo_code": [f"JV-25-{i:05d}" for i in range(n_ref + 1, n_ref + n_cur + 1)],
        "mjo_date": tgl_cur,
        "acc_id": rng.choice([1, 2, 3, 4, 5, 6], n_cur, p=[0.10, 0.10, 0.35, 0.25, 0.15, 0.05]),
        "of_id": rng.integers(1, 5, n_cur),
        "djo_debit": np.round(rng.lognormal(15.1, 0.75, n_cur), 0),  # Pergeseran rata-rata & sebaran!
        "input_by": rng.choice(["siti", "budi", "andi", "rina", "pegawai_baru"], n_cur)
    })
    bad_cur = rng.choice(n_cur, int(n_cur * 0.03), replace=False)
    cur_df.loc[bad_cur, "djo_debit"] *= rng.uniform(35, 65, len(bad_cur))
    cur_df["is_anomaly"] = 0
    cur_df.loc[bad_cur, "is_anomaly"] = 1

    # Gabungkan dengan master akun & fitur turunan
    for name, df in [("reference_journal", ref_df), ("current_journal", cur_df)]:
        df = df.merge(acc[["acc_id", "acc_code", "acc_name"]], on="acc_id", how="left")
        df["log_amt"] = np.log1p(df["djo_debit"])
        df["dow"] = df["mjo_date"].dt.dayofweek
        df["month"] = df["mjo_date"].dt.month
        df.to_parquet(DATA_DIR / f"{name}.parquet", index=False)

    anggaran_df.to_parquet(DATA_DIR / "anggaran_biaya.parquet", index=False)
    print(f"✅ Dataset ACCIS siap di folder '{DATA_DIR}':")
    print(f"   - reference_journal.parquet : {len(ref_df)} baris (Jan-Sep 2025)")
    print(f"   - current_journal.parquet   : {len(cur_df)} baris (Okt-Des 2025, mengandung Data Drift)")
    print(f"   - anggaran_biaya.parquet    : {len(anggaran_df)} baris (Pagu 12 bulan)")

if __name__ == "__main__":
    build_accis_datasets()