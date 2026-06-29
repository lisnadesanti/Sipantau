import warnings
import mysql.connector
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error

warnings.filterwarnings("ignore")


def get_koneksi():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="sipsn-prediction"
    )


def ambil_data_nasional(uji_tahun):
    conn = get_koneksi()

    query = f"""
    SELECT tahun, SUM(jml_timbulan_tahun) AS total_timbulan
    FROM data_sipsn
    WHERE uji_tahun = {uji_tahun}
    GROUP BY tahun
    ORDER BY tahun
    """

    df = pd.read_sql(query, conn)
    conn.close()

    df["tahun"] = df["tahun"].astype(int)
    df["total_timbulan"] = pd.to_numeric(
        df["total_timbulan"],
        errors="coerce"
    )

    return df


def cari_model_terbaik(series):
    best_aic = float("inf")
    best_order = None
    best_model = None

    # grid kecil dulu karena data sedikit
    for p in range(0, 3):
        for d in range(0, 3):
            for q in range(0, 3):
                try:
                    model = ARIMA(series, order=(p, d, q))
                    fitted = model.fit()

                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                        best_model = fitted
                except Exception:
                    continue

    return best_order, best_model, best_aic


def evaluasi_prediksi_2025():
    # data latih: histori sampai 2024
    df_train = ambil_data_nasional(2025)

    # data aktual 2025: diambil dari snapshot uji_tahun 2026
    df_actual = ambil_data_nasional(2026)

    train_series = df_train["total_timbulan"]

    # ambil nilai aktual tahun 2025
    actual_2025 = df_actual.loc[
        df_actual["tahun"] == 2025,
        "total_timbulan"
    ].values[0]

    best_order, best_model, best_aic = cari_model_terbaik(train_series)

    pred_2025 = best_model.forecast(steps=1).iloc[0]

    mape = mean_absolute_percentage_error(
        [actual_2025],
        [pred_2025]
    ) * 100

    print("=" * 60)
    print("EVALUASI MODEL NASIONAL")
    print("=" * 60)
    print("Data latih (uji_tahun=2025):")
    print(df_train)
    print("-" * 60)
    print(f"Model terbaik: ARIMA{best_order}")
    print(f"AIC terbaik   : {best_aic:.4f}")
    print(f"Prediksi 2025 : {pred_2025:.4f}")
    print(f"Aktual 2025   : {actual_2025:.4f}")
    print(f"MAPE          : {mape:.4f}%")
    print("=" * 60)

    return best_order, mape


def prediksi_final_2026():
    # histori sampai 2025
    df_final = ambil_data_nasional(2026)
    final_series = df_final["total_timbulan"]

    best_order, best_model, best_aic = cari_model_terbaik(final_series)

    pred_2026 = best_model.forecast(steps=1).iloc[0]

    print("=" * 60)
    print("PREDIKSI FINAL NASIONAL 2026")
    print("=" * 60)
    print(df_final)
    print("-" * 60)
    print(f"Model terbaik: ARIMA{best_order}")
    print(f"AIC terbaik   : {best_aic:.4f}")
    print(f"Prediksi 2026 : {pred_2026:.4f}")
    print("=" * 60)

    return best_order, pred_2026


if __name__ == "__main__":
    evaluasi_prediksi_2025()
    prediksi_final_2026()
