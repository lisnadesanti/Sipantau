import os
import warnings
import mysql.connector
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

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


def klasifikasi_error(ape):
    if ape < 10:
        return "Sangat Akurat"
    if ape < 20:
        return "Akurat/Bagus"
    if ape < 50:
        return "Layak/Cukup"
    return "Tidak Akurat/Buruk"


def evaluasi_rolling_nasional():
    hasil = []

    # 4 baris evaluasi akurasi
    daftar_uji = [2022, 2023, 2024, 2025]

    for tahun_uji in daftar_uji:
        print("=" * 70)
        print(f"SEDANG MEMPROSES EVALUASI TAHUN UJI {tahun_uji}")
        print("=" * 70)

        df_train = ambil_data_nasional(tahun_uji)
        df_aktual = ambil_data_nasional(tahun_uji + 1)

        if df_train.empty:
            print(f"Data training untuk {tahun_uji} kosong.")
            continue

        aktual_row = df_aktual[df_aktual["tahun"] == tahun_uji]

        if aktual_row.empty:
            print(f"Data aktual tahun {tahun_uji} tidak ditemukan.")
            continue

        train_series = df_train["total_timbulan"]

        best_order, best_model, best_aic = cari_model_terbaik(
            train_series
        )

        if best_model is None:
            print(f"Model gagal ditemukan untuk tahun uji {tahun_uji}.")
            continue

        prediksi = float(best_model.forecast(steps=1).iloc[0])
        aktual = float(aktual_row["total_timbulan"].iloc[0])

        error_absolut = abs(aktual - prediksi)

        if aktual != 0:
            persen_error = (error_absolut / aktual) * 100
        else:
            persen_error = 0.0

        kategori = klasifikasi_error(persen_error)

        training_awal = int(df_train["tahun"].min())
        training_akhir = int(df_train["tahun"].max())

        hasil.append(
            {
                "Kab/Kota": "Nasional",
                "Variabel": "jml_timbulan_tahun",
                "Data Training": f"{training_awal}-{training_akhir}",
                "Tahun Uji": tahun_uji,
                "Orde ARIMA": f"ARIMA{best_order}",
                "AIC": round(float(best_aic), 4),
                "Hasil Prediksi": round(prediksi, 4),
                "Testing": round(aktual, 4),
                "Error Absolut": round(error_absolut, 4),
                "Persen Error": round(persen_error, 4),
                "Kategori": kategori,
            }
        )

        print(f"Training       : {training_awal}-{training_akhir}")
        print(f"Model Terbaik  : ARIMA{best_order}")
        print(f"AIC            : {best_aic:.4f}")
        print(f"Prediksi       : {prediksi:.4f}")
        print(f"Testing        : {aktual:.4f}")
        print(f"Error Absolut  : {error_absolut:.4f}")
        print(f"Persen Error   : {persen_error:.4f}%")
        print(f"Kategori       : {kategori}")

    # 1 baris prediksi final 2026
    print("=" * 70)
    print("SEDANG MEMPROSES PREDIKSI FINAL 2026")
    print("=" * 70)

    df_final = ambil_data_nasional(2026)

    if not df_final.empty:
        final_series = df_final["total_timbulan"]
        best_order, best_model, best_aic = cari_model_terbaik(
            final_series
        )

        if best_model is not None:
            prediksi_2026 = float(best_model.forecast(steps=1).iloc[0])

            training_awal = int(df_final["tahun"].min())
            training_akhir = int(df_final["tahun"].max())

            hasil.append(
                {
                    "Kab/Kota": "Nasional",
                    "Variabel": "jml_timbulan_tahun",
                    "Data Training": f"{training_awal}-{training_akhir}",
                    "Tahun Uji": 2026,
                    "Orde ARIMA": f"ARIMA{best_order}",
                    "AIC": round(float(best_aic), 4),
                    "Hasil Prediksi": round(prediksi_2026, 4),
                    "Testing": "-",
                    "Error Absolut": "-",
                    "Persen Error": "-",
                    "Kategori": "Prediksi Final",
                }
            )

            print(f"Training       : {training_awal}-{training_akhir}")
            print(f"Model Terbaik  : ARIMA{best_order}")
            print(f"AIC            : {best_aic:.4f}")
            print(f"Prediksi 2026  : {prediksi_2026:.4f}")

    df_hasil = pd.DataFrame(hasil)

    if df_hasil.empty:
        print("\nTidak ada hasil evaluasi yang berhasil dibuat.")
        return

    # rata-rata error hanya untuk baris evaluasi, bukan prediksi final
    df_eval_only = df_hasil[df_hasil["Kategori"] != "Prediksi Final"].copy()
    df_eval_only["Persen Error"] = pd.to_numeric(
        df_eval_only["Persen Error"],
        errors="coerce"
    )

    rata_rata_error = df_eval_only["Persen Error"].mean()

    ringkasan = pd.DataFrame(
        [
            {
                "Kab/Kota": "Nasional",
                "Variabel": "jml_timbulan_tahun",
                "Jumlah Pengujian": len(df_eval_only),
                "Rata-rata Persen Error": round(rata_rata_error, 4),
                "Kategori Umum": klasifikasi_error(rata_rata_error),
            }
        ]
    )

    os.makedirs("output", exist_ok=True)

    file_excel = os.path.join(
        "output",
        "tabel_evaluasi_nasional.xlsx"
    )
    file_csv = os.path.join(
        "output",
        "tabel_evaluasi_nasional.csv"
    )
    file_ringkasan = os.path.join(
        "output",
        "ringkasan_evaluasi_nasional.xlsx"
    )

    df_hasil.to_excel(file_excel, index=False)
    df_hasil.to_csv(file_csv, index=False)
    ringkasan.to_excel(file_ringkasan, index=False)

    print("\n" + "=" * 70)
    print("HASIL AKHIR EVALUASI NASIONAL")
    print("=" * 70)
    print(df_hasil)
    print("\nRata-rata Persen Error:", round(rata_rata_error, 4))
    print("Kategori Umum:", klasifikasi_error(rata_rata_error))
    print("\nFile berhasil dibuat:")
    print(f"- {file_excel}")
    print(f"- {file_csv}")
    print(f"- {file_ringkasan}")


if __name__ == "__main__":
    evaluasi_rolling_nasional()
