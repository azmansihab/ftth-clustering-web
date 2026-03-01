import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import folium
from streamlit_folium import st_folium
import fiona
import zipfile
import os
import matplotlib.colors as mcolors

# Mengabaikan warning bawaan
import warnings
warnings.filterwarnings('ignore')

# Mengaktifkan driver KML
fiona.drvsupport.supported_drivers['KML'] = 'rw'
fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'

st.set_page_config(layout="wide", page_title="FTTH ODP Clustering (Rectangle Grid)")

st.title("🗺️ FTTH Clustering: Batas Rectangle (Grid Spasial)")
st.markdown("Membagi titik homepass menggunakan sapuan Timur-Barat & Utara-Selatan untuk menghasilkan batas **Persegi Panjang (Rectangle)** yang sangat rapi, konsisten, dan terarah.")

# --- FUNGSI PENDUKUNG ---

def process_spatial_file(file):
    """Membaca data KML/KMZ dan mengambil koordinat"""
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
    
    if points_only.crs is None:
        points_only.set_crs(epsg=4326, inplace=True)
    else:
        points_only.to_crs(epsg=4326, inplace=True)
        
    return pd.DataFrame({
        'id': range(1, len(points_only) + 1), 
        'lon': points_only.geometry.x, 
        'lat': points_only.geometry.y
    })

def create_rectangular_boundaries(df_points, cluster_col='final_odp_id'):
    """Membuat batas Bounding Box (Rectangle) untuk setiap ODP"""
    gdf_points = gpd.GeoDataFrame(
        df_points, geometry=gpd.points_from_xy(df_points.lon, df_points.lat), crs="EPSG:4326"
    )
    
    polygons = []
    for odp_id, group in gdf_points.groupby(cluster_col):
        # Mengambil semua titik dalam 1 ODP
        points_union = group.geometry.unary_union
        
        # .envelope secara otomatis membuat bentuk Rectangle sempurna (Bounding Box)
        # Kita beri buffer (tambahan jarak) sangat kecil agar titik tidak persis di garis tepi
        rect = points_union.envelope.buffer(0.00015).envelope 
        
        polygons.append({'odp_id': odp_id, 'geometry': rect})
        
    gdf_boundaries = gpd.GeoDataFrame(polygons, crs="EPSG:4326")
    return gdf_boundaries, gdf_points

# --- UI UTAMA STREAMLIT ---

st.sidebar.header("Pengaturan Jaringan")
max_capacity = st.sidebar.number_input("Kapasitas Maks per ODP", min_value=4, max_value=64, value=20)
# Chunk size ini akan menentukan "Lebar Pita" sapuan vertikal dari kanan ke kiri
chunk_size = st.sidebar.number_input("Lebar Sapuan Kolom (Titik)", min_value=50, max_value=1000, value=200)
opacity_slider = st.sidebar.slider("Transparansi Warna Area", 0.1, 1.0, 0.4)

uploaded_file = st.file_uploader("Unggah File KML/KMZ (Titik Homepass)", type=['kml', 'kmz'])

if uploaded_file is not None:
    with st.spinner('Membaca data file spasial...'):
        df_hp = process_spatial_file(uploaded_file)
        
        if st.button("Mulai Proses Boundary Rectangle (Kanan ke Kiri)"):
            
            with st.spinner('Langkah 1: Slicing Grid Spasial (Super Cepat)...'):
                # 1. SAPUAN SUMBU X (TIMUR KE BARAT)
                # Urutkan rumah dari Kanan (Timur) ke Kiri (Barat)
                df_hp = df_hp.sort_values(by='lon', ascending=False).reset_index(drop=True)
                
                # Potong menjadi beberapa kolom vertikal
                n_cols = max(1, len(df_hp) // chunk_size)
                if n_cols > 1:
                    df_hp['col_id'] = pd.qcut(df_hp.index, q=n_cols, labels=False, duplicates='drop')
                else:
                    df_hp['col_id'] = 0
                
                df_hp['final_odp_id'] = -1
                global_odp_counter = 0
                
                # 2. SAPUAN SUMBU Y (UTARA KE SELATAN) DI DALAM SETIAP KOLOM
                for col_id in sorted(df_hp['col_id'].unique()):
                    mask = df_hp['col_id'] == col_id
                    col_data = df_hp[mask]
                    if len(col_data) == 0: continue
                    
                    # Urutkan dari Atas (Utara) ke Bawah (Selatan)
                    col_data = col_data.sort_values(by='lat', ascending=False).reset_index()
                    
                    # Potong tegas tepat per `max_capacity` (misal 20 titik)
                    # Ini memastikan isi ODP pas 20 dan bentuknya mengotak rapi
                    col_data['micro_id'] = col_data.index // max_capacity
                    
                    # Simpan ID final
                    df_hp.loc[col_data['index'], 'final_odp_id'] = col_data['micro_id'] + global_odp_counter
                    global_odp_counter += col_data['micro_id'].nunique()

            with st.spinner('Langkah 2: Membentuk Geometri Bounding Box...'):
                gdf_boundaries, gdf_points_all = create_rectangular_boundaries(df_hp)

            with st.spinner('Langkah 3: Merender Peta Satelit...'):
                base_colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
                def get_color(odp_id): return base_colors[int(odp_id) % len(base_colors)]

                center_lat, center_lon = df_hp['lat'].mean(), df_hp['lon'].mean()
                m = folium.Map(
                    location=[center_lat, center_lon], 
                    zoom_start=16, 
                    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', 
                    attr='Esri'
                )
                
                # Render Area (Rectangle)
                folium.GeoJson(
                    gdf_boundaries,
                    name="Boundary ODP (Rectangle)",
                    style_function=lambda f: {
                        'fillColor': get_color(f['properties']['odp_id']),
                        'color': 'white', # Garis tepi putih tegas
                        'weight': 2,
                        'fillOpacity': opacity_slider
                    },
                    tooltip=folium.GeoJsonTooltip(fields=['odp_id'], aliases=['ODP ID:'])
                ).add_to(m)

                # Render Titik Rumah
                for _, row in df_hp.iterrows():
                    folium.CircleMarker(
                        location=[row['lat'], row['lon']], 
                        radius=3,
                        color='black', weight=1, fill=True, fillColor='#00ff00', fillOpacity=1,
                        popup=f"ODP: {row['final_odp_id']}"
                    ).add_to(m)

                st.subheader(f"Selesai: {global_odp_counter} ODP Kotak Terbentuk")
                
                # --- FITUR DOWNLOAD ---
                st.markdown("### 📥 Unduh Hasil Desain (Export)")
                col1, col2 = st.columns(2)
                
                with col1:
                    csv_data = df_hp.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📄 Unduh Data Homepass (CSV)",
                        data=csv_data,
                        file_name="hasil_clustering_homepass.csv",
                        mime="text/csv"
                    )
                    
                with col2:
                    kml_file_path = "hasil_boundary_rectangle.kml"
                    if os.path.exists(kml_file_path):
                        os.remove(kml_file_path)
                        
                    gdf_boundaries_export = gdf_boundaries.to_crs(epsg=4326)
                    gdf_boundaries_export.to_file(kml_file_path, driver='KML')
                    
                    with open(kml_file_path, "rb") as kml_file:
                        st.download_button(
                            label="🗺️ Unduh Boundary Kotak (KML)",
                            data=kml_file,
                            file_name="hasil_boundary_rectangle.kml",
                            mime="application/vnd.google-earth.kml+xml"
                        )

                st_folium(m, width=1200, height=700, returned_objects=[])

else:
    st.info("Unggah KML/KMZ Anda untuk memulai.")