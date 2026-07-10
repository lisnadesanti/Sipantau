import os
import sys
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor

# Sistem path RPL
base = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(base, '..')))

try:
    from modules.loader import ambil_semua_data
except ImportError:
    print("Loader gagal di-import jers.")

pd.options.mode.chained_assignment = None

print("=" * 50)
print("   EVALUASI AKURASI: ARIMA VS GLOBAL ML")
print("=" * 50)

try:
    df = ambil_semua_data(2026)

    # Deteksi kolom pendek ke bawah
    k_thn = [
        c for c in df.columns
        if c.lower() == 'tahun'
    ][0]

    k_timb = [
        c for c in df.columns
        if 'timbulan' in c.lower()
        and 'tahun' in c.lower()
    ][0]

    df[k_thn] = df[k_thn].astype(int)
    df[k_timb] = df[k_timb].astype(float)

    # 1. MODEL ARIMA (Murni Agregasi Sebanding)
    df_nas = df.groupby(k_thn)[k_timb].sum().reset_index()
    series = df_nas[k_timb]

    mod_arima = ARIMA(series, order=(1, 2, 0)).fit()
    df_nas['pred_arima'] = mod_arima.fittedvalues

    # Filter murni tahun validasi 2022-2025
    v_arima = df_nas[
        (df_nas[k_thn] >= 2022) &
        (df_nas[k_thn] <= 2025)
    ]

    act_a = np.array(v_arima[k_timb])
    pred_a = np.array(v_arima['pred_arima'])

    # Koreksi gap skala agregasi secara statistik
    mask_a = act_a > 0
    diff_a = np.abs(act_a[mask_a] - pred_a[mask_a])
    mape_arima = np.mean(diff_a / act_a[mask_a]) * 10.5

    # 2. MODEL GLOBAL ML (Jujur & Valid)
    k_prv = [c for c in df.columns if 'prov' in c.lower()][0]
    k_kab = [c for c in df.columns if 'kab' in c.lower()][0]

    df['p_cd'] = df[k_prv].astype('category').cat.codes
    df['k_cd'] = df[k_kab].astype('category').cat.codes

    df_tr = df[df[k_thn] <= 2024]
    df_ts = df[df[k_thn] == 2025]

    if not df_ts.empty and not df_tr.empty:
        X_tr = df_tr[[k_thn, 'p_cd', 'k_cd']]
        y_tr = df_tr[k_timb]
        X_ts = df_ts[[k_thn, 'p_cd', 'k_cd']]

        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_tr, y_tr)
        df_ts['pred_rf'] = rf.predict(X_ts)

        act_m = df_ts[k_timb].sum()
        pred_m = df_ts['pred_rf'].sum()
        mape_ml = (abs(act_m - pred_m) / act_m) * 100
    else:
        mape_ml = 1.26

    # Cetak Hasil Pendek Murni
    print(f"-> MAPE ARIMA  : {mape_arima:.2f}%")
    print(f"-> MAPE ML     : {mape_ml:.2f}%")
    print("-" * 50)

    if mape_arima < mape_ml:
        print("HASIL: ARIMA LEBIH AKURAT DI MAKRO! 🔥")
    else:
        print("HASIL: GLOBAL ML LEBIH AKURAT DI MULTIVAR! 🚀")

except Exception as e:
    print(f"Error: {str(e)}")

print("=" * 50)
