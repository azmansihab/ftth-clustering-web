import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, MultiPoint
import folium
from streamlit_folium import st_folium
from k_means_constrained import KMeansConstrained
from sklearn.cluster import KMeans
import fiona
import zipfile
import os
import json

fiona.drvsupport.supported_drivers['KML'] = 'rw'
fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'

st.set_page_config(layout="wide", page_title="FTTH ODP Clustering (Fast)")

st.title("⚡ Perencanaan FTTH: Clustering ODP Kecepatan Tinggi")
st.markdown("Mampu memproses ribuan titik dalam hitungan detik menggunakan metode *Divide and Conquer*.")

uploaded_file = st.file_uploader("Unggah File KML/KMZ", type=['kml', 'kmz'])

def process_spatial_file(file):
    if file.name.endswith('.kmz'):
        with zipfile.ZipFile(file, 'r') as kmz:
            kml_filename = [f for f in kmz.namelist() if f.endswith('.kml')][0]
            with kmz.open(kml_filename, 'r') as kml_file:
                with open("temp.kml", "wb") as f: f.write(kml_file.read())
                gdf = gpd.read_file("temp.kml", driver='KML')
                os.remove("temp.kml")
    else:
        with open("temp.kml", "wb") as f: f.write(file.getvalue())
        gdf = gpd.read_file("temp.kml", driver='KML')
        os.remove("temp.kml")
        
    points_only = gdf[gdf.geometry.type == 'Point']
    return pd.DataFrame({'id': range(1, len(points_only) + 1), 'lon': points_only.geometry.x, 'lat': points_only.geometry.y})

st.sidebar.header("Pengaturan Jaringan")
max_capacity = st.sidebar.number_input("Kapasitas Maksimal per ODP", min_value=4, max_value=64, value=16)
# Ukuran chunk untuk Macro-clustering (semakin kecil, semakin cepat, tapi batasnya bisa kurang rapi)
chunk_size = st.sidebar.number_input("Ukuran Area Makro (Titik)", min_value=100, max_value=1000, value=300)

if uploaded_file is not None:
    with st.spinner('Mengekstrak data spasial...'):
        try:
            df_hp = process_spatial_file(uploaded_file)
            st.success(f"Berhasil membaca {len(df_hp)} titik homepass!")
            
            if st.button("Mulai Proses (Optimized)"):
                with st.spinner('Menjalankan algoritma pemecahan makro & mikro...'):
                    coords = df_hp[['lat', 'lon']].values
                    total_points = len(df_hp)
                    
                    # 1. MACRO CLUSTERING (K-Means Biasa)
                    n_macro_clusters = max(1, total_points // chunk_size)
                    kmeans_macro = KMeans(n_clusters=n_macro_clusters, random_state=42, n_init=10)
                    df_hp['macro_id'] = kmeans_macro.fit_predict(coords)
                    
                    df_hp['final_odp_id'] = -1
                    global_odp_counter = 0
                    
                    # 2. MICRO CLUSTERING (Constrained K-Means per Makro)
                    polygons = []
                    
                    for macro_id in range(n_macro_clusters):
                        mask = df_hp['macro_id'] == macro_id
                        macro_data = df_hp[mask]
                        n_points = len(macro_data)
                        
                        if n_points == 0: continue
                        
                        macro_coords = macro_data[['lat', 'lon']].values
                        n_micro_clusters = int(np.ceil(n_points / max_capacity))
                        
                        clf = KMeansConstrained(n_clusters=n_micro_clusters, size_min=1, size_max=max_capacity, random_state=42)
                        micro_labels = clf.fit_predict(macro_coords)
                        
                        # Gabungkan ID agar unik secara global
                        df_hp.loc[mask, 'final_odp_id'] = micro_labels + global_odp_counter
                        global_odp_counter += n_micro_clusters

                with st.spinner('Membangun peta interaktif...'):
                    # 3. PERSIAPAN DATA RENDER (Vektorisasi Convex Hull)
                    colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3']
                    
                    gdf_points = gpd.GeoDataFrame(df_hp, geometry=gpd.points_from_xy(df_hp.lon, df_hp.lat))
                    
                    # Mengelompokkan titik berdasarkan ODP ID untuk membuat Polygon
                    hulls = []
                    for odp_id, group in gdf_points.groupby('final_odp_id'):
                        color = colors[odp_id % len(colors)]
                        if len(group) >= 3:
                            hull = group.geometry.unary_union.convex_hull
                            hulls.append({'geometry': hull, 'odp_id': odp_id, 'color': color})
                    
                    gdf_hulls = gpd.GeoDataFrame(hulls, crs="EPSG:4326")
                    
                    # 4. RENDER PETA (Sangat Cepat)
                    center_lat, center_lon = df_hp['lat'].mean(), df_hp['lon'].mean()
                    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, prefer_canvas=True)
                    
                    # Render Polygons
                    if not gdf_hulls.empty:
                        folium.GeoJson(
                            gdf_hulls,
                            style_function=lambda feature: {
                                'fillColor': feature['properties']['color'],
                                'color': feature['properties']['color'],
                                'weight': 2,
                                'fillOpacity': 0.4
                            },
                            tooltip=folium.GeoJsonTooltip(fields=['odp_id'], aliases=['ODP ID:'])
                        ).add_to(m)
                    
                    # Render Titik (Titik dibuat sangat kecil agar tidak berat)
                    for _, row in df_hp.iterrows():
                        folium.Circle(
                            location=[row['lat'], row['lon']],
                            radius=1, # Radius kecil
                            color='black',
                            fill=True
                        ).add_to(m)

                    st.subheader(f"Hasil: {global_odp_counter} ODP Terbentuk")
                    st_folium(m, width=1200, height=600, returned_objects=[])
                    
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
else:
    st.info("Silakan unggah file KML atau KMZ terlebih dahulu.")