import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from modules.loader import ambil_semua_data
from modules.preprocessing import siapkan_deret_waktu

# Mengatur agar plot menggunakan gaya visual yang bersih
if "seaborn-v0_8-whitegrid" in plt.style.available:
    plt.style.use("seaborn-v0_8-whitegrid")
else:
    plt.style.use("default")


def plot_arima_identification(series):
    """Menampilkan plot ACF dan PACF dari data deret waktu.

    Digunakan untuk mengidentifikasi kandidat parameter p dan q
    sesuai prosedur Box-Jenkins di Bab 3 dan Bab 4 Laporan Skripsi.
    """
    if series.empty or len(series) < 3:
        print("[WARNING] Data terlalu pendek untuk menghitung ACF/PACF.")
        return

    # Sesuai Bab 4, d=1 digunakan karena data asli memiliki tren kuat
    series_diff = series.diff().dropna()

    # Membuat 2 grafik berdampingan sesuai Gambar 4.1 di Bab 4
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot ACF untuk menentukan nilai q (Moving Average)
    plot_acf(series_diff, ax=ax1, lags=min(10, len(series_diff) - 1))
    ax1.set_title('Grafik ACF (Menentukan nilai q)')
    ax1.set_xlabel('Lag')
    ax1.set_ylabel('Korelasi')

    # Plot PACF untuk menentukan nilai p (Autoregressive)
    plot_pacf(series_diff, ax=ax2, lags=min(10, len(series_diff) - 1),
              method='ywm')
    ax2.set_title('Grafik PACF (Menentukan nilai p)')
    ax2.set_xlabel('Lag')
    ax2.set_ylabel('Korelasi Parsial')

    plt.tight_layout()
    print("\n[INFO] Menampilkan grafik ACF dan PACF.")
    print("Silakan simpan grafik ini untuk Gambar 4.1 di Laporan Skripsi!")
    plt.show()


if __name__ == "__main__":
    print("=" * 50)
    print("PROSES IDENTIFIKASI PARAMETER P DAN Q")
    print("=" * 50)

    try:
        df_raw = ambil_semua_data(2026)
        series_ts = siapkan_deret_waktu(df_raw, wilayah="Nasional")

        if not series_ts.empty:
            plot_arima_identification(series_ts)
        else:
            print("[ERROR] Deret waktu kosong setelah pra-pemrosesan.")
    except Exception as e:
        print(f"[ERROR] Terjadi kegagalan pemrosesan data: {e}")
