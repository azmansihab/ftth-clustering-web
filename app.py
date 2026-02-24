import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, MultiPoint
from shapely.ops import voronoi_diagram
import folium
from streamlit_folium import st_folium
from k_means_constrained import KMeansConstrained
from sklearn.cluster import KMeans
import fiona
import zipfile
import os
import matplotlib.colors as mcolors
import osmnx as ox
import warnings

# Mengabaikan warning bawaan pandas/shapely agar terminal tetap bersih
warnings.filterwarnings('ignore')

# Mengaktifkan driver untuk membaca KML dan KMZ
fiona.drvsupport.supported_drivers['KML'] = 'rw'
fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'

# Konfigurasi Halaman Web
st.set_page_config(layout="wide", page_title="FTTH ODP Clustering (Road Bounds)")

st.title("🗺️ FTTH Clustering: Batas Rapi Mengikuti Jalan")
st.markdown("Menggunakan Voronoi yang dipotong otomatis oleh jaringan jalan (OpenStreetMap) sehingga batas ODP rapi dan tidak menyeberang jalan.")

# --- FUNGSI PENDUKUNG ---

def process_spatial_file(file):
    """Fungsi untuk mengekstrak titik dari KML/KMZ"""
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
    
    # Standarisasi ke sistem koordinat GPS (WGS84 / EPSG:4326)
    if points_only.crs is None:
        points_only.set_crs(epsg=4326, inplace=True)
    else:
        points_only.to_crs(epsg=4326, inplace=True)
        
    return pd.DataFrame({
        'id': range(1, len(points_only) + 1), 
        'lon': points_only.geometry.x, 
        'lat': points_only.geometry.y
    })

def get_road_network(gdf_points):
    """Men-download jaringan jalan dari OSM berdasarkan area koordinat rumah"""
    west, south, east, north = gdf_points.total_bounds
    buffer_deg = 0.002 # Tambah buffer area sekitar 200 meter
    bbox = (west - buffer_deg, south - buffer_deg, east + buffer_deg, north + buffer_deg)
    
    try:
        # Menarik data jalan dari OpenStreetMap
        roads = ox.features_from_bbox(bbox=bbox, tags={'highway': True})
        return roads
    except Exception as e:
        st.warning("Gagal mengunduh jalan dari OSM. Pastikan ada koneksi internet. Batas akan menggunakan Voronoi biasa.")
        return None

def create_physical_boundaries(df_points, cluster_col='final_odp_id'):
    """Fungsi inti: Membuat Voronoi lalu memotongnya dengan jalan (Road Cutter)"""
    gdf_points = gpd.GeoDataFrame(
        df_points, 
        geometry=gpd.points_from_xy(df_points.lon, df_points.lat), 
        crs="EPSG:4326"
    )
    
    # 1. Hitung Centroid (Titik Tengah tiap ODP)
    centroids_data = []
    for cluster_id, group in df_points.groupby(cluster_col):
        centroids_data.append({
            'odp_id': cluster_id, 
            'geometry': Point(group['lon'].mean(), group['lat'].mean())
        })
    gdf_centroids = gpd.GeoDataFrame(centroids_data, crs="EPSG:4326")

    # 2. Buat Poligon Voronoi Dasar
    mask_polygon = gdf_points.unary_union.convex_hull.buffer(0.001)
    gdf_mask = gpd.GeoDataFrame(geometry=[mask_polygon], crs="EPSG:4326")
    
    centroids_multipoint = MultiPoint(gdf_centroids['geometry'].tolist())
    voronoi_raw = voronoi_diagram(centroids_multipoint, envelope=mask_polygon.envelope)
    gdf_voronoi_raw = gpd.GeoDataFrame(geometry=list(voronoi_raw.geoms), crs="EPSG:4326")
    
    # Beri ID ODP pada masing-masing Poligon Voronoi
    gdf_voronoi = gpd.sjoin(gdf_voronoi_raw, gdf_centroids, how="inner", predicate="contains")
    gdf_voronoi = gpd.clip(gdf_voronoi, gdf_mask)

    # 3. PROSES PEMOTONGAN JALAN (ROAD CUTTER) YANG SUDAH DIPERBAIKI
    roads_gdf = get_road_network(gdf_points)
    
    if roads_gdf is not None and not roads_gdf.empty:
        # Ubah ke koordinat Meter (Web Mercator) agar ketebalan jalan akurat
        gdf_voronoi_m = gdf_voronoi.to_crs(epsg=3857)
        roads_m = roads_gdf.to_crs(epsg=3857)
        gdf_points_m = gdf_points.to_crs(epsg=3857)
        
        # Beri ketebalan jalan 3 meter sebagai "pisau potong"
        road_buffer = roads_m.geometry.buffer(3) 
        road_union = road_buffer.unary_union
        
        # Copy struktur data agar ID tidak hilang (Solusi KeyError)
        gdf_voronoi_cut = gdf_voronoi_m.copy()
        
        # Lakukan pemotongan (Difference) hanya pada kolom geometry
        gdf_voronoi_cut['geometry'] = gdf_voronoi_m.geometry.difference(road_union)
        
        # Buang hasil potong yang kebetulan kosong (is_empty)
        gdf_voronoi_cut = gdf_voronoi_cut[~gdf_voronoi_cut.geometry.is_empty]
        
        # Pecah poligon yang terbelah dua oleh jalan menjadi baris terpisah
        gdf_exploded = gdf_voronoi_cut.explode(index_parts=False).reset_index(drop=True)
        
        # Filter: Hanya simpan area potongan yang benar-benar ada rumahnya
        final_polygons = []
        for _, odp_area in gdf_exploded.iterrows():
            poly = odp_area.geometry 
            odp_id = odp_area['odp_id']
            
            # Cek apakah titik homepass ODP ini masuk ke dalam poligon potongan ini
            points_in_this_odp = gdf_points_m[gdf_points_m[cluster_col] == odp_id]
            if not points_in_this_odp.empty and points_in_this_odp.geometry.within(poly).any():
                final_polygons.append({'odp_id': odp_id, 'geometry': poly})
                
        if final_polygons:
            gdf_final = gpd.GeoDataFrame(final_polygons, crs="EPSG:3857")
            # Kembalikan ke koordinat GPS (WGS84)
            return gdf_final.to_crs(epsg=4326), gdf_points
            
    return gdf_voronoi, gdf_points

# --- UI UTAMA STREAMLIT ---

st.sidebar.header("Pengaturan Jaringan")
max_capacity = st.sidebar.number_input("Kapasitas Maks per ODP", min_value=4, max_value=64, value=16)
chunk_size = st.sidebar.number_input("Ukuran Area Makro (Kecepatan)", min_value=100, max_value=2000, value=500)
opacity_slider = st.sidebar.slider("Transparansi Warna", 0.1, 1.0, 0.4)

uploaded_file = st.file_uploader("Unggah File KML/KMZ (Titik Homepass)", type=['kml', 'kmz'])

if uploaded_file is not None:
    with st.spinner('Membaca data file spasial...'):
        df_hp = process_spatial_file(uploaded_file)
        st.success(f"Berhasil memuat {len(df_hp)} titik homepass!")
        
        if st.button("Mulai Proses Boundary Fisik (Satelit)"):
            
            # 1. K-Means Constrained (Super Cepat dengan Makro-Mikro)
            with st.spinner('Langkah 1: Menghitung klaster maksimal 16 titik...'):
                coords = df_hp[['lat', 'lon']].values
                n_macro = max(1, len(df_hp) // chunk_size)
                df_hp['macro_id'] = KMeans(n_clusters=n_macro, random_state=42).fit_predict(coords)
                
                df_hp['final_odp_id'] = -1
                global_odp_counter = 0
                
                for macro_id in range(n_macro):
                    mask = df_hp['macro_id'] == macro_id
                    macro_data = df_hp[mask]
                    if len(macro_data) == 0: continue
                    
                    n_micro = max(1, int(np.ceil(len(macro_data) / max_capacity)))
                    clf = KMeansConstrained(n_clusters=n_micro, size_min=1, size_max=max_capacity, random_state=42)
                    df_hp.loc[mask, 'final_odp_id'] = clf.fit_predict(macro_data[['lat', 'lon']].values) + global_odp_counter
                    global_odp_counter += n_micro

            # 2. Proses Geometri Pemotong Jalan
            with st.spinner('Langkah 2: Download jalan OSM & memotong batas area (Mohon tunggu 1-2 menit)...'):
                gdf_boundaries, gdf_points_all = create_physical_boundaries(df_hp)

            # 3. Render Peta dengan Basemap Satelit
            with st.spinner('Langkah 3: Merender Peta Satelit...'):
                # Palet warna yang bervariasi
                base_colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
                def get_color(odp_id): return base_colors[int(odp_id) % len(base_colors)]

                center_lat, center_lon = df_hp['lat'].mean(), df_hp['lon'].mean()
                
                # Menggunakan tile ESRI World Imagery agar mirip Google Earth
                m = folium.Map(
                    location=[center_lat, center_lon], 
                    zoom_start=16, 
                    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', 
                    attr='Esri'
                )
                
                # Render Poligon Area
                folium.GeoJson(
                    gdf_boundaries,
                    name="Boundary ODP",
                    style_function=lambda f: {
                        'fillColor': get_color(f['properties']['odp_id']),
                        'color': 'white', # Garis pinggir putih agar kontras di peta satelit
                        'weight': 1.5,
                        'fillOpacity': opacity_slider
                    },
                    tooltip=folium.GeoJsonTooltip(fields=['odp_id'], aliases=['ODP ID:'])
                ).add_to(m)

                # Render Titik Rumah
                for _, row in df_hp.iterrows():
                    folium.CircleMarker(
                        location=[row['lat'], row['lon']], 
                        radius=3,
                        color='black', 
                        weight=1, 
                        fill=True, 
                        fillColor='#ffcc00', # Titik warna kuning terang
                        fillOpacity=1,
                        popup=f"ODP: {row['final_odp_id']}"
                    ).add_to(m)

                st.subheader(f"Hasil Akhir: {global_odp_counter} ODP (Terpotong oleh Jalan Fisik)")
                st_folium(m, width=1200, height=700, returned_objects=[])

else:
    st.info("Silakan unggah file KML atau KMZ Anda untuk memulai.")