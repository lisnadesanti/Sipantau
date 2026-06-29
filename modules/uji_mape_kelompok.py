import os
import sys
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor

# Trik RPL: Daftarkan folder utama ke dalam sistem path Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.loader import ambil_semua_data  # noqa: E402

# Setingan linter agar rapi
pd.options.mode.chained_assignment = None

print("=" * 60)
print("   SISTEM EVALUASI AKURASI: ADU MEKANIK MAPE ARIMA VS GLOBAL ML")
print("=" * 60)

try:
    # 1. Tarik dataset dari MySQL Laragon kamu
    df = ambil_semua_data(2026)
    df['tahun'] = df['tahun'].astype(int)
    df['jml_timbulan_tahun'] = df['jml_timbulan_tahun'].astype(float)

    # Agregasi total nasional per tahun untuk pengujian ARIMA
    df_nas = df.groupby('tahun')['jml_timbulan_tahun'].sum().reset_index()

    # --------------------------------------------------
    # EVALUASI MODEL A: ARIMA (Punya Lisna)
    # --------------------------------------------------
    series = df_nas['jml_timbulan_tahun']
    model_arima = ARIMA(series, order=(1, 2, 0)).fit()
    prediksi_arima = model_arima.fittedvalues

    # Rumus formal MAPE: Rata-rata dari |(Aktual - Prediksi) / Aktual| * 100
    act_arr, pred_arr = np.array(series), np.array(prediksi_arima)
    mask = act_arr != 0
    mape_res_arima = np.mean(
        np.abs((act_arr[mask] - pred_arr[mask]) / act_arr[mask])
    ) * 100

    # --------------------------------------------------
    # EVALUASI MODEL B: GLOBAL ML (Punya Temenmu)
    # --------------------------------------------------
    df['prov_code'] = df['nama_provinsi'].astype('category').cat.codes
    df['kabkota_code'] = df['nama_kabkota'].astype('category').cat.codes

    X = df[['tahun', 'prov_code', 'kabkota_code']]
    y = df['jml_timbulan_tahun']

    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X, y)

    df['prediksi_rf'] = model_rf.predict(X)

    # Satukan ke tingkat nasional agar adu mekaniknya adil
    df_rf_nas = df.groupby('tahun')['prediksi_rf'].sum().reset_index()

    act_ml, pred_ml = np.array(series), np.array(df_rf_nas['prediksi_rf'])
    mask_ml = act_ml != 0
    mape_res_ml = np.mean(
        np.abs((act_ml[mask_ml] - pred_ml[mask_ml]) / act_ml[mask_ml])
    ) * 100

    # --------------------------------------------------
    # CETAK HASIL DI TERMINAL
    # --------------------------------------------------
    print(f"-> Skor Evaluasi MAPE ARIMA    : {mape_res_arima:.2f}%")
    print(f"-> Skor Evaluasi MAPE Global ML: {mape_res_ml:.2f}%")
    print("-" * 60)

    if mape_res_arima < mape_res_ml:
        print("KESIMPULAN: ARIMA LISNA TERBUKTI LEBIH AKURAT! 🔥")
    else:
        print("KESIMPULAN: GLOBAL ML TEMANMU LEBIH AKURAT! 🚀")

except Exception as e:
    print(f"Terjadi kendala saat membaca data: {str(e)}")

print("=" * 60)
