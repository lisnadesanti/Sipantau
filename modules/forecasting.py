import warnings
from statsmodels.tsa.arima.model import ARIMA
from modules.loader import ambil_semua_data
from modules.preprocessing import siapkan_deret_waktu

# Mengabaikan pesan warning teknis agar tabel di terminal tetap rapi
warnings.filterwarnings("ignore")


def cari_model_terbaik(series):
    """Mencari kombinasi p, d, q terbaik berdasarkan nilai AIC terkecil.

    Menerapkan metode Grid Search 27 iterasi sesuai prosedur yang ditulis
    pada Laporan Skripsi Bab 4 Tabel 4.3.
    """
    best_aic = float("inf")
    best_order = None
    best_model = None

    # Header Tabel yang Rapi (Dipotong agar tidak melanggar batas kolom)
    hdr = f"{'No':<4} | {'Model ARIMA':<15} | {'Nilai AIC':<20} | {'Status'}"
    print(f"\n{hdr}")
    print("-" * 55)

    count = 1
    # Loop 3x3x3 = 27 Iterasi
    for p in range(0, 3):
        for d in range(0, 3):
            for q in range(0, 3):
                model_name = f"ARIMA({p},{d},{q})"
                try:
                    # Proses Fitting
                    model = ARIMA(series, order=(p, d, q))
                    fitted = model.fit()
                    current_aic = fitted.aic

                    # Logika Penentuan Status
                    if current_aic < best_aic:
                        best_aic = current_aic
                        best_order = (p, d, q)
                        best_model = fitted
                        status = "Terbaik Sementara"
                    else:
                        status = "-"

                    # Cetak Baris Berhasil
                    print(f"{count:<4} | {model_name:<15} | "
                          f"{current_aic:<20.4f} | {status}")

                except (ValueError, IndexError):
                    # Cetak Baris Gagal (Tetap muncul nomornya)
                    print(f"{count:<4} | {model_name:<15} | "
                          f"{'Gagal/Error':<20} | Skip")

                count += 1

    return best_order, best_model, best_aic


if __name__ == "__main__":
    print("=" * 55)
    print("PROSES EVALUASI 27 KANDIDAT MODEL (GRID SEARCH)")
    print("=" * 55)

    try:
        df_raw = ambil_semua_data(2026)
        series_ts = siapkan_deret_waktu(df_raw, wilayah="Nasional")

        if not series_ts.empty:
            b_order, b_model, b_aic = cari_model_terbaik(series_ts)
            print("-" * 55)
            print(f"KESIMPULAN FINAL: Model Terbaik adalah ARIMA{b_order}")
            print(f"Nilai AIC Terendah: {b_aic:.4f}")
        else:
            print("Data deret waktu tidak ditemukan atau kosong.")
    except Exception as e:
        print(f"[ERROR] Gagal melakukan kalkulasi grid search: {e}")
