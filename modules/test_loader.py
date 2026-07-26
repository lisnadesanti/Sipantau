from loader import ambil_data_nasional  # Mengimpor fungsi dari loader.py

# Ambil data untuk tahun 2022
df_train = ambil_data_nasional(2022)

# Menampilkan data yang diambil
print(df_train)
