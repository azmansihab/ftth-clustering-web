import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, MultiPoint
import folium
from streamlit_folium import st_folium
from k_means_constrained import KMeansConstrained
import fiona
import zipfile
import os

# Aktifkan driver KML di Fiona (karena defaultnya mati)
fiona.drvsupport.supported_drivers['KML'] = 'rw'
fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'

st.set_page_config(layout="wide", page_title="FTTH ODP Clustering")

st.title("🌐 Perencanaan Jaringan FTTH: Clustering ODP")
st.markdown("Unggah file KML/KMZ (Maksimal 1GB) berisi titik homepass. Sistem akan mengelompokkan maksimal 16 rumah dan membuat batas *coverage* (Convex Hull).")

# Fitur Upload File
uploaded_file = st.file_uploader("Unggah File KML atau KMZ", type=['kml', 'kmz'])

def process_spatial_file(file):
    """Fungsi untuk mengekstrak data dari KML atau KMZ"""
    if file.name.endswith('.kmz'):
        # KMZ sebenarnya adalah KML yang di-zip. Kita harus ekstrak dulu.
        with zipfile.ZipFile(file, 'r') as kmz:
            # Cari file kml di dalam zip
            kml_filename = [f for f in kmz.namelist() if f.endswith('.kml')][0]
            with kmz.open(kml_filename, 'r') as kml_file:
                # Simpan sementara untuk dibaca GeoPandas
                with open("temp.kml", "wb") as f:
                    f.write(kml_file.read())
                gdf = gpd.read_file("temp.kml", driver='KML')
                os.remove("temp.kml") # Hapus file sementara
    else:
        # Jika KML, langsung baca
        with open("temp.kml", "wb") as f:
            f.write(file.getvalue())
        gdf = gpd.read_file("temp.kml", driver='KML')
        os.remove("temp.kml")
        
    # Ambil koordinat Lat dan Lon dari titik (Point)
    # Kita filter hanya yang berupa "Point" (mengabaikan poligon/garis jalan jika ada di KML)
    points_only = gdf[gdf.geometry.type == 'Point']
    
    df = pd.DataFrame({
        'id': range(1, len(points_only) + 1),
        'lon': points_only.geometry.x,
        'lat': points_only.geometry.y
    })
    return df

st.sidebar.header("Pengaturan Jaringan")
max_capacity = st.sidebar.number_input("Kapasitas Maksimal per ODP", min_value=4, max_value=32, value=16)

# Logika Utama
if uploaded_file is not None:
    with st.spinner('Mengekstrak data dari file spasial...'):
        try:
            df_homepass = process_spatial_file(uploaded_file)
            st.success(f"Berhasil membaca {len(df_homepass)} titik homepass dari file!")
            
            if st.button("Mulai Proses Clustering & Boundary"):
                with st.spinner('Sedang menghitung jarak dan membuat cluster (Convex Hull)...'):
                    # Proses Constrained K-Means
                    X = df_homepass[['lat', 'lon']].values
                    n_clusters = int(np.ceil(len(df_homepass) / max_capacity))
                    
                    clf = KMeansConstrained(
                        n_clusters=n_clusters,
                        size_min=1,
                        size_max=max_capacity,
                        random_state=42
                    )
                    clf.fit_predict(X)
                    df_homepass['cluster_id'] = clf.labels_
                    
                    # Titik tengah untuk memulai peta
                    center_lat = df_homepass['lat'].mean()
                    center_lon = df_homepass['lon'].mean()
                    m = folium.Map(location=[center_lat, center_lon], zoom_start=16)
                    
                    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple']
                    
                    for cluster_id in df_homepass['cluster_id'].unique():
                        cluster_data = df_homepass[df_homepass['cluster_id'] == cluster_id]
                        cluster_color = colors[cluster_id % len(colors)]
                        
                        points = []
                        for _, row in cluster_data.iterrows():
                            folium.CircleMarker(
                                location=[row['lat'], row['lon']],
                                radius=4,
                                color=cluster_color,
                                fill=True,
                                popup=f"ODP: {cluster_id}"
                            ).add_to(m)
                            points.append(Point(row['lon'], row['lat']))
                        
                        # Convex Hull (Boundary Coverage)
                        if len(points) >= 3:
                            multipoint = MultiPoint(points)
                            convex_hull = multipoint.convex_hull
                            sim_geo = gpd.GeoSeries(convex_hull).simplify(tolerance=0.0001)
                            
                            folium.GeoJson(
                                data=sim_geo.to_json(),
                                style_function=lambda x, color=cluster_color: {
                                    'fillColor': color, 'color': color, 'weight': 2, 'fillOpacity': 0.2
                                }
                            ).add_to(m)

                    st.subheader("Peta Hasil Desain FTTH")
                    st_folium(m, width=1200, height=600)
                    
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file: {e}. Pastikan file berisi data titik (Point).")
else:
    st.info("Silakan unggah file KML atau KMZ terlebih dahulu di kotak atas.")