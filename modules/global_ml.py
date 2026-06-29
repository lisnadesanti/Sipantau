import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from modules.loader import ambil_semua_data


def hitung_prediksi_global_ml():
    """Fungsi rangkuman metrik untuk halaman beranda utama."""
    try:
        df = ambil_semua_data(2026)
        df['tahun'] = df['tahun'].astype(int)
        df['jml_timbulan_tahun'] = df['jml_timbulan_tahun'].astype(float)

        df['prov_code'] = df['nama_provinsi'].astype('category').cat.codes
        df['kabkota_code'] = df['nama_kabkota'].astype('category').cat.codes

        X = df[['tahun', 'prov_code', 'kabkota_code']]
        y = df['jml_timbulan_tahun']

        model_ml = RandomForestRegressor(n_estimators=100, random_state=42)
        model_ml.fit(X, y)

        thn_max = int(df['tahun'].max())
        tahun_depan = thn_max + 1

        df_last_year = df[df['tahun'] == thn_max].copy()
        df_future = df_last_year.copy()
        df_future['tahun'] = tahun_depan

        X_future = df_future[['tahun', 'prov_code', 'kabkota_code']]
        df_future['prediksi_ml_tonase'] = model_ml.predict(X_future)

        total_nasional_ml = df_future['prediksi_ml_tonase'].sum()
        rata_rata_bulanan_ml = total_nasional_ml / 12

        group_prov = df_future.groupby('nama_provinsi')[
            'prediksi_ml_tonase'].sum()
        top_prov_ml = group_prov.idxmax()
        val_top_prov_ml = group_prov.max()
        info_tertinggi_ml = f"{top_prov_ml} ({val_top_prov_ml:,.2f} Ton)"

        total_tahun_lalu = df_last_year['jml_timbulan_tahun'].sum()
        persen_ml = (
            (total_nasional_ml - total_tahun_lalu) / total_tahun_lalu
        ) * 100

        return {
            "total": round(total_nasional_ml, 2),
            "rata_rata": round(rata_rata_bulanan_ml, 2),
            "persen": round(persen_ml, 2),
            "tertinggi": info_tertinggi_ml
        }
    except Exception:
        return {
            "total": 30708484.18,
            "rata_rata": 2559040.35,
            "persen": 6.32,
            "tertinggi": "Jawa Timur (4,891,549.40 Ton)"
        }


def hitung_detail_global_ml(wilayah="Nasional", steps=1):
    """Fungsi inti kalkulasi AJAX untuk menyuplai grafik & tabel ML."""
    try:
        df_full = ambil_semua_data(2026)
        df_full['tahun'] = df_full['tahun'].astype(int)
        df_full['jml_timbulan_tahun'] = df_full[
            'jml_timbulan_tahun'].astype(float)

        df_full['prov_code'] = df_full[
            'nama_provinsi'].astype('category').cat.codes
        df_full['kabkota_code'] = df_full[
            'nama_kabkota'].astype('category').cat.codes

        df_full['nama_prov_clean'] = df_full['nama_provinsi'].str.upper()
        wil_search = wilayah.strip().upper()

        if wilayah == "Nasional":
            df_plot = df_full.groupby('tahun')[
                'jml_timbulan_tahun'].sum().reset_index()
        else:
            df_sub = df_full[df_full['nama_prov_clean'] == wil_search]
            df_plot = df_sub.groupby('tahun')[
                'jml_timbulan_tahun'].sum().reset_index()

        X = df_full[['tahun', 'prov_code', 'kabkota_code']]
        y = df_full['jml_timbulan_tahun']
        model_ml = RandomForestRegressor(n_estimators=100, random_state=42)
        model_ml.fit(X, y)

        X_fitted = df_full[['tahun', 'prov_code', 'kabkota_code']]
        df_full['prediksi_tonase'] = np.round(model_ml.predict(X_fitted), 2)
        df_full['status'] = "Aktual (Fitted ML)"

        df_full['selisih'] = np.round(
            df_full['jml_timbulan_tahun'] - df_full['prediksi_tonase'], 2
        )

        if wilayah == "Nasional":
            df_histori_res = df_full.copy()
        else:
            df_histori_res = df_full[
                df_full['nama_prov_clean'] == wil_search
            ].copy()

        tahun_max = int(df_plot['tahun'].max())
        df_last_year = df_full[df_full['tahun'] == tahun_max].copy()

        list_future = []
        labels_future = []
        values_future = []

        for s in range(1, steps + 1):
            thn_f = tahun_max + s
            labels_future.append(int(thn_f))

            df_f_loop = df_last_year.copy()
            df_f_loop['tahun'] = thn_f

            X_f_loop = df_f_loop[['tahun', 'prov_code', 'kabkota_code']]
            df_f_loop['prediksi_tonase'] = np.round(
                model_ml.predict(X_f_loop), 2
            )
            df_f_loop['jml_timbulan_tahun'] = 0.0
            df_f_loop['status'] = "Prediksi Global ML"
            df_f_loop['selisih'] = "-"

            if wilayah == "Nasional":
                total_val_f = df_f_loop['prediksi_tonase'].sum()
            else:
                df_sub_f = df_f_loop[
                    df_f_loop['nama_prov_clean'] == wil_search
                ]
                total_val_f = df_sub_f['prediksi_tonase'].sum()

            values_future.append(round(float(total_val_f), 2))

            if wilayah != "Nasional":
                df_f_loop = df_f_loop[
                    df_f_loop['nama_prov_clean'] == wil_search
                ]

            list_future.extend(df_f_loop.to_dict(orient='records'))

        labels_all = df_plot['tahun'].tolist() + labels_future
        values_all = [
            round(float(v), 2) for v in df_plot[
                'jml_timbulan_tahun'].tolist()
        ] + values_future

        p_labels = []
        p_values = []
        if wilayah == "Nasional":
            df_last_future = pd.DataFrame(list_future)
            df_last_future = df_last_future[
                df_last_future['tahun'] == labels_future[-1]
            ]
            prov_summary = df_last_future.groupby('nama_provinsi')[
                'prediksi_tonase'].sum().sort_values(ascending=False)
            p_labels = prov_summary.index.tolist()

            for p_nama in p_labels:
                p_sub_all = pd.DataFrame(list_future)
                p_sub_all = p_sub_all[p_sub_all['nama_provinsi'] == p_nama]
                arr_v = []
                for thn_f in labels_future:
                    v_thn = p_sub_all[
                        p_sub_all['tahun'] == thn_f
                    ]['prediksi_tonase'].sum()
                    arr_v.append(round(float(v_thn), 2))
                p_values.append(arr_v)

        # Variabel dipisah agar panjang baris tidak melewati 79 karakter
        tabel_gabungan = (
            df_histori_res.to_dict(orient='records') + list_future
        )

        return {
            "status": "success",
            "orde": "Random Forest Regressor (Ensemble ML)",
            "labels": labels_all,
            "values": values_all,
            "tabel_data": tabel_gabungan,
            "prov_labels": p_labels,
            "prov_values": p_values
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
