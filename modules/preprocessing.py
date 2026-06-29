import warnings
import pandas as pd

# Abaikan warning dari pandas agar tampilan terminal rapi
warnings.filterwarnings("ignore")


def bersihkan_timbulan_nol(df):
    """Menangani nilai timbulan sampah yang bernilai 0 atau negatif.

    Sesuai Bab 4 Skripsi, nilai 0 mencerminkan kegagalan input data
    di tingkat daerah, sehingga dikonversi atau difilter.
    """
    if df.empty:
        return df

    df_clean = df.copy()
    # Memastikan kolom tahunan bertipe numerik
    df_clean["jml_timbulan_tahun"] = pd.to_numeric(
        df_clean["jml_timbulan_tahun"], errors="coerce"
    )
    # Filter data yang hanya di atas 0
    df_clean = df_clean[df_clean["jml_timbulan_tahun"] > 0]
    return df_clean


def hitung_differencing(series, order=1):
    """Menghitung selisih data periode saat ini dengan periode sebelumnya.

    Sesuai persamaan transformasi (1.2) di Bab 2 untuk membuat
    data deret waktu mencapai kondisi stasioner.
    """
    if order == 0:
        return series

    diff_series = series.copy()
    for _ in range(order):
        diff_series = diff_series.diff().dropna()
    return diff_series


def siapkan_deret_waktu(df, wilayah="Nasional"):
    """Mengubah data tabular SIPSN menjadi deret waktu kronologis tahunan.

    Melakukan sinkronisasi dan agregat sum berdasarkan variabel tahun.
    """
    df_clean = bersihkan_timbulan_nol(df)
    if df_clean.empty:
        return pd.Series()

    if wilayah != "Nasional":
        df_clean["nama_prov_clean"] = df_clean["nama_provinsi"].str.upper()
        df_clean = df_clean[df_clean["nama_prov_clean"] == wilayah.upper()]

    # Grouping berdasarkan tahun untuk deret waktu tunggal ARIMA
    df_ts = df_clean.groupby("tahun")["jml_timbulan_tahun"].sum().reset_index()
    df_ts = df_ts.sort_values("tahun")

    # Mengembalikan dalam bentuk pandas Series dengan index tahun
    return pd.Series(
        data=df_ts["jml_timbulan_tahun"].values,
        index=df_ts["tahun"].values
    )


if __name__ == "__main__":
    print("=" * 50)
    print("SISTEM CHECKER PREPROCESSING.PY (PEP8 SYNC)")
    print("=" * 50)

    # Simulasi data dummy historis SIPSN 2019-2025
    data_dummy = {
        "tahun": [2019, 2020, 2021, 2022, 2023, 2024, 2025],
        "nama_provinsi": ["JAWA BARAT"] * 7,
        "jml_timbulan_tahun": [100, 120, 0, 150, 170, 190, 210]
    }
    df_test = pd.DataFrame(data_dummy)

    print("[...] Menjalankan pembersihan nilai 0...")
    df_filtered = bersihkan_timbulan_nol(df_test)
    print(f"[OK] Baris sebelum: {len(df_test)}, setelah: {len(df_filtered)}")

    print("[...] Mengubah ke format time series...")
    series_ts = siapkan_deret_waktu(df_test, wilayah="Nasional")
    print(f"[OK] Deret waktu terbentuk:\n{series_ts}")

    print("[...] Mencoba hitung differencing orde 1...")
    series_diff = hitung_differencing(series_ts, order=1)
    print(f"[OK] Hasil diff orde 1:\n{series_diff}")
    print("=" * 50)
