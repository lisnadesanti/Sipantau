import warnings
import mysql.connector
import pandas as pd

# Abaikan warning dari pandas agar tampilan terminal rapi
warnings.filterwarnings("ignore")


def get_koneksi():
    """Mengatur koneksi ke database MySQL"""
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="sipsn-prediction"
    )


def ambil_data_nasional(uji_tahun):
    """KHUSUS ARIMA: Mengambil total timbulan sampah nasional per tahun.

    Digunakan untuk melatih model ARIMA (Data yang sudah di-SUM).
    """
    conn = get_koneksi()
    query = f"""
    SELECT tahun, SUM(jml_timbulan_tahun) AS total_timbulan
    FROM data_sipsn
    WHERE uji_tahun = {uji_tahun}
    GROUP BY tahun
    ORDER BY tahun ASC
    """
    df = pd.read_sql(query, conn)
    conn.close()

    if not df.empty:
        df["tahun"] = df["tahun"].astype(int)
        df["total_timbulan"] = pd.to_numeric(
            df["total_timbulan"], errors="coerce"
        )
    return df


def ambil_semua_data(uji_tahun):
    """KHUSUS TABEL: Mengambil detail semua provinsi dan kabupaten/kota.

    Digunakan untuk menampilkan tabel panjang (3306 data) di dashboard.
    """
    conn = get_koneksi()
    query = f"""
    SELECT tahun, nama_provinsi, nama_kabkota,
           jml_timbulan_harian, jml_timbulan_tahun
    FROM data_sipsn
    WHERE uji_tahun = {uji_tahun}
    ORDER BY nama_provinsi ASC, nama_kabkota ASC, tahun ASC
    """
    df = pd.read_sql(query, conn)
    conn.close()

    if not df.empty:
        df["tahun"] = df["tahun"].astype(int)
        df["jml_timbulan_tahun"] = pd.to_numeric(
            df["jml_timbulan_tahun"], errors="coerce"
        )
    return df


def ambil_daftar_provinsi(uji_tahun):
    """KHUSUS DROPDOWN: Mengambil daftar nama provinsi unik dari database."""
    conn = get_koneksi()
    query = f"""
    SELECT DISTINCT nama_provinsi
    FROM data_sipsn
    WHERE uji_tahun = {uji_tahun}
    ORDER BY nama_provinsi
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df["nama_provinsi"].tolist()


def simpan_hasil_prediksi(tahun, wilayah, nilai_prediksi, orde):
    """Menyimpan hasil peramalan ARIMA ke tabel 'hasil_prediksi'.

    Nama kolom disesuaikan dengan screenshot phpMyAdmin:
    - tahun_prediksi
    - nama_kabkota (untuk menyimpan nama wilayah/provinsi)
    - hasil_prediksi
    - orde_arima
    """
    conn = get_koneksi()
    cursor = conn.cursor()

    query = """
    INSERT INTO hasil_prediksi (tahun_prediksi, nama_kabkota,
                                hasil_prediksi, orde_arima)
    VALUES (%s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE hasil_prediksi = %s, orde_arima = %s
    """

    nilai_fix = round(float(nilai_prediksi), 2)
    val = (tahun, wilayah, nilai_fix, orde, nilai_fix, orde)

    try:
        cursor.execute(query, val)
        conn.commit()
        print(f"[OK] Data {wilayah} tahun {tahun} berhasil tersimpan.")
    except mysql.connector.Error as err:
        print(f"[DATABASE ERROR] Gagal menyimpan: {err}")
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    print("=" * 50)
    print("SISTEM CHECKER LOADER.PY (FINAL SYNC)")
    print("=" * 50)

    try:
        tahun_uji = 2026

        detail = ambil_semua_data(tahun_uji)
        print(f"[OK] Berhasil menarik {len(detail)} baris data detail.")

        print("[...] Mencoba sinkronisasi simpan ke tabel hasil_prediksi...")
        simpan_hasil_prediksi(
            2026, "DKI JAKARTA", 3456789.12, "ARIMA(1,2,0)"
        )

        print("\nSTATUS: LOADER SIAP DAN SINKRON")
        print("=" * 50)

    except Exception as e:
        print(f"\n[ERROR] Terjadi kesalahan: {e}")
