import os
import warnings
import pandas as pd
from modules.loader import ambil_semua_data
from modules.preprocessing import siapkan_deret_waktu
from modules.forecasting import cari_model_terbaik

# Mengabaikan pesan peringatan agar terminal bersih
warnings.filterwarnings("ignore")


def klasifikasi_error(ape):
    """Mengklasifikasikan kualitas performa peramalan.

    Sesuai dengan standar kriteria pengujian MAPE yang ditulis
    pada Laporan Skripsi Bab 2 Bagian Klasifikasi Kualitas Prediksi.
    """
    if ape < 10:
        return "Sangat Akurat"
    if ape < 20:
        return "Akurat/Bagus"
    if ape < 50:
        return "Layak/Cukup"
    return "Tidak Akurat/Buruk"


def evaluasi_rolling_nasional():
    """Menjalankan pengujian validasi rolling forecast (2022-2025).

    Menghitung rumus galat persentase absolut (MAPE) secara kumulatif
    sebagai komponen pengujian utama pada Laporan Skripsi Bab 4.
    """
    hasil = []
    daftar_uji = [2022, 2023, 2024, 2025]

    for tahun_uji in daftar_uji:
        print(f"\n>>> MEMPROSES EVALUASI TAHUN: {tahun_uji}")
        df_train_raw = ambil_semua_data(tahun_uji)
        df_aktual_raw = ambil_semua_data(tahun_uji + 1)

        if df_train_raw.empty or df_aktual_raw.empty:
            print(f"Data {tahun_uji} tidak lengkap.")
            continue

        train_series = siapkan_deret_waktu(df_train_raw, wilayah="Nasional")
        df_ak = siapkan_deret_waktu(df_aktual_raw, wilayah="Nasional")

        if train_series.empty or df_ak.empty:
            continue

        # Mengambil nilai aktual pada index tahun uji
        if tahun_uji in df_ak.index:
            aktual = float(df_ak.loc[tahun_uji])
        else:
            continue

        best_order, best_model, _ = cari_model_terbaik(train_series)

        if best_model is not None:
            prediksi = float(best_model.forecast(steps=1).iloc[0])
            error_absolut = abs(aktual - prediksi)
            persen_error = (error_absolut / aktual) * 100 \
                if aktual != 0 else 0

            kategori = klasifikasi_error(persen_error)
            hasil.append({
                "Tahun Uji": tahun_uji,
                "Orde ARIMA": f"ARIMA{best_order}",
                "Hasil Prediksi": round(prediksi, 2),
                "Data Aktual": round(aktual, 2),
                "Persen Error (%)": round(persen_error, 2),
                "Kategori": kategori
            })

    print("\n>>> MEMPROSES PREDIKSI FINAL 2026")
    df_final_raw = ambil_semua_data(2026)
    series_final = siapkan_deret_waktu(df_final_raw, wilayah="Nasional")

    if not series_final.empty:
        best_order, best_model, _ = cari_model_terbaik(series_final)
        prediksi_2026 = float(best_model.forecast(steps=1).iloc[0])
        hasil.append({
            "Tahun Uji": 2026,
            "Orde ARIMA": f"ARIMA{best_order}",
            "Hasil Prediksi": round(prediksi_2026, 2),
            "Data Aktual": "-",
            "Persen Error (%)": "-",
            "Kategori": "Prediksi Final"
        })

    df_hasil = pd.DataFrame(hasil)

    # Simpan Hasil ke Folder Output di luar modules
    path_output = os.path.join("output")
    os.makedirs(path_output, exist_ok=True)
    df_hasil.to_excel(
        os.path.join(path_output, "hasil_skripsi.xlsx"), index=False
    )

    print("\n" + "=" * 40)
    print(df_hasil.to_string(index=False))
    print("=" * 40)

    # Hitung rata-rata MAPE sesuai persamaan rumus (1.4) di Bab 2
    eval_data = df_hasil[df_hasil["Kategori"] != "Prediksi Final"]
    if not eval_data.empty:
        mape = eval_data["Persen Error (%)"].mean()
        print(f"RATA-RATA MAPE: {round(mape, 2)}%")
        print(f"KESIMPULAN: {klasifikasi_error(mape)}")


if __name__ == "__main__":
    evaluasi_rolling_nasional()
