import os
import re
import glob
import pandas as pd
import mysql.connector

# folder tempat semua file Excel
FOLDER_DATASET = r"E:\KULIAH\SEMESTER 8\SKRIPSI\Dataset"

# koneksi database MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="sipsn-prediction"
)

cursor = conn.cursor()

# hapus isi tabel dulu supaya tidak dobel saat import ulang
cursor.execute("DELETE FROM data_sipsn")
conn.commit()

sql_insert = """
INSERT INTO data_sipsn (
    tahun,
    periode,
    uji_tahun,
    nama_provinsi,
    nama_kabkota,
    jml_timbulan_harian,
    jml_timbulan_tahun,
    sumber_file
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
"""

total_inserted = 0

# ambil semua file excel
file_list = glob.glob(os.path.join(FOLDER_DATASET, "*.xlsx"))

for file_path in file_list:
    file_name = os.path.basename(file_path)

    # ambil angka tahun uji dari nama file, misal "Data Uji 2022"
    match = re.search(r"Data Uji\s*(\d{4})", file_name, re.IGNORECASE)
    if not match:
        print(f"Lewati file, format nama tidak cocok: {file_name}")
        continue

    uji_tahun = int(match.group(1))

    # baca file excel
    df = pd.read_excel(file_path)

    # rapikan nama kolom
    df.columns = [col.strip().lower() for col in df.columns]

    kolom_wajib = [
        "tahun",
        "periode",
        "nama_provinsi",
        "nama_kabkota",
        "jml_timbulan_harian",
        "jml_timbulan_tahun",
    ]

    # cek kolom wajib
    kolom_tidak_ada = [col for col in kolom_wajib if col not in df.columns]
    if kolom_tidak_ada:
        print(f"Kolom tidak ditemukan di {file_name}: {kolom_tidak_ada}")
        continue

    # ambil kolom yang dibutuhkan
    df = df[kolom_wajib].copy()

    # hapus baris kosong penting
    df = df.dropna(
        subset=[
            "tahun",
            "periode",
            "nama_provinsi",
            "nama_kabkota",
            "jml_timbulan_harian",
            "jml_timbulan_tahun",
        ]
    )

    # ubah tipe data
    df["tahun"] = df["tahun"].astype(int)
    df["periode"] = df["periode"].astype(int)
    df["jml_timbulan_harian"] = pd.to_numeric(
        df["jml_timbulan_harian"],
        errors="coerce"
    )
    df["jml_timbulan_tahun"] = pd.to_numeric(
        df["jml_timbulan_tahun"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["jml_timbulan_harian", "jml_timbulan_tahun"]
    )

    data_to_insert = [
        (
            int(row["tahun"]),
            int(row["periode"]),
            uji_tahun,
            str(row["nama_provinsi"]),
            str(row["nama_kabkota"]),
            float(row["jml_timbulan_harian"]),
            float(row["jml_timbulan_tahun"]),
            file_name,
        )
        for _, row in df.iterrows()
    ]

    cursor.executemany(sql_insert, data_to_insert)
    conn.commit()

    total_inserted += len(data_to_insert)
    print(f"{file_name} -> {len(data_to_insert)} baris masuk")

print(f"\nTotal data berhasil diimport: {total_inserted}")

cursor.close()
conn.close()
