import warnings
from mysql.connector import connect
from statsmodels.tsa.arima.model import ARIMA
from modules.loader import ambil_semua_data, ambil_daftar_provinsi
from modules.forecasting import cari_model_terbaik
from modules.global_ml import hitung_detail_global_ml

warnings.filterwarnings("ignore")


def jalankan_seeder_sekali_saja():
    """Skrip seeder untuk menghitung semua data prediksi Nasional & Provinsi

    secara permanen dan menyimpannya ke database cukup sekali saja.
    """
    print("====== MEMULAI PROSES DATA SEEDING PERMANEN ======")

    # 1. KONEKSI DATABASE
    try:
        db = connect(
            host="localhost",
            user="root",
            password="",
            database="sipsn-prediction"
        )
        cursor = db.cursor()
    except Exception as e:
        print(f"[ERROR] Gagal koneksi database: {str(e)}")
        return

    # 2. BERSIHKAN TABEL SEBELUM DIISI DATA FINAL
    print("Memperbarui status tabel...")
    cursor.execute("TRUNCATE TABLE hasil_prediksi")
    cursor.execute("TRUNCATE TABLE hasil_prediksi_ml")
    db.commit()

    # 3. AMBIL DAFTAR WILAYAH (Nasional + Semua Provinsi)
    list_wilayah = ["Nasional"]
    try:
        provinsi_db = ambil_daftar_provinsi(2026)
        list_wilayah.extend(provinsi_db)
    except Exception as e:
        print(f"[WARNING] Gagal ambil daftar provinsi: {str(e)}")

    print(f"Total wilayah ditemukan untuk dihitung: {len(list_wilayah)}")

    # 4. LOOPING PROSES HITUNG & SEEDING UNTUK MODEL ARIMA
    print("\n[1/2] Memproses Perhitungan Algoritma ARIMA...")
    try:
        df_full = ambil_semua_data(2026)
        df_full['tahun'] = df_full['tahun'].astype(int)
        df_full['nama_prov_clean'] = df_full['nama_provinsi'].str.upper()

        for wilayah in list_wilayah:
            wil_search = wilayah.strip().upper()

            if wilayah == "Nasional":
                df_plot = df_full.groupby('tahun')['jml_timbulan_tahun'].sum()
                df_plot = df_plot.reset_index()
                best_order = (1, 2, 0)
                best_model = ARIMA(df_plot['jml_timbulan_tahun'],
                                   order=best_order).fit()
            else:
                df_histori = df_full[df_full['nama_prov_clean'] == wil_search]
                if df_histori.empty:
                    continue
                df_plot = df_histori.groupby('tahun')[
                    'jml_timbulan_tahun'].sum().reset_index()
                best_order, best_model, _ = cari_model_terbaik(
                    df_plot['jml_timbulan_tahun']
                )

            orde_str = f"ARIMA{best_order}"
            forecast_values = best_model.forecast(steps=2)
            tahun_max = int(df_plot['tahun'].max())

            for idx, val_f in enumerate(forecast_values):
                tahun_target = tahun_max + (idx + 1)
                cursor.execute(
                    """INSERT INTO hasil_prediksi
                    (nama_kabkota, tahun_prediksi, hasil_prediksi, orde_arima)
                    VALUES (%s, %s, %s, %s)""",
                    (wilayah, tahun_target, round(float(val_f), 2), orde_str)
                )
        db.commit()
        print("-> Sukses mengunci seluruh data prediksi ARIMA ke database!")
    except Exception as e:
        print(f"[ERROR] Gagal memproses data ARIMA: {str(e)}")

    # 5. LOOPING PROSES HITUNG & SEEDING UNTUK MODEL MACHINE LEARNING (FIXED)
    print("\n[2/2] Memproses Perhitungan Model Machine Learning...")
    try:
        for wilayah in list_wilayah:
            # Memanggil fungsi hitung ML kelompok untuk tahun 2026 & 2027
            res_ml = hitung_detail_global_ml(wilayah=wilayah, steps=2)

            for row in res_ml.get("tabel_data", []):
                tahun_target = int(row.get("tahun", 2026))

                # Perbaikan Filter: Memastikan baris data masa depan
                # dari hasil hitung ML masuk, baik skala Nasional maupun
                if tahun_target > 2025:
                    pred_tonase = float(row.get("prediksi_tonase", 0))

                    cursor.execute(
                        """INSERT INTO hasil_prediksi_ml
                        (nama_kabkota, tahun_prediksi, hasil_prediksi, mape)
                        VALUES (%s, %s, %s, %s)""",
                        (wilayah, tahun_target, pred_tonase, 1.26)
                    )
        db.commit()
        print("-> Sukses mengunci seluruh data prediksi ML ke database!")
    except Exception as e:
        print(f"[ERROR] Gagal memproses data ML: {str(e)}")

    # 6. TUTUP KONEKSI
    cursor.close()
    db.close()
    print("\n====== SEEDER SELESAI! DATABASE SUDAH TERKUNCI SEMPURNA ======")


if __name__ == "__main__":
    jalankan_seeder_sekali_saja()
