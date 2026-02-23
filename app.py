import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, MultiPoint, Polygon
from shapely.ops import voronoi_diagram
import folium
from streamlit_folium import st_folium
from k_means_constrained import KMeansConstrained
from sklearn.cluster import KMeans
import fiona
import zipfile
import os
import matplotlib.colors as mcolors

# Konfigurasi Driver KML
fiona.drvsupport.supported_drivers['KML'] = 'rw'
fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'

st.set_page_config(layout="wide", page_title="FTTH ODP Clustering (Rapi)")

st.title("🗺️ Perencanaan FTTH: Clustering ODP Rapi (Voronoi)")
st.markdown("Menggunakan algoritma *Voronoi Tessellation* untuk membuat batas coverage yang rapi, lurus, dan tanpa tumpang tindih seperti peta profesional.")

# --- FUNGSI PENDUKUNG ---
def process_spatial_file(file):
    """Membaca KML/KMZ dan mengambil titik homepass"""
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
    # Pastikan menggunakan CRS (Sistem Koordinat) standar WGS84
    if points_only.crs is None:
        points_only.set_crs(epsg=4326, inplace=True)
    else:
        points_only.to_crs(epsg=4326, inplace=True)
        
    return pd.DataFrame({'id': range(1, len(points_only) + 1), 'lon': points_only.geometry.x, 'lat': points_only.geometry.y})

def create_voronoi_boundaries(df_points, cluster_col='final_odp_id'):
    """
    Fungsi inti untuk membuat batas Voronoi yang rapi dari hasil clustering.
    """
    # 1. Buat GeoDataFrame dari semua titik
    gdf_points = gpd.GeoDataFrame(df_points, geometry=gpd.points_from_xy(df_points.lon, df_points.lat), crs="EPSG:4326")

    # 2. Hitung Centroid (Titik Tengah) dari setiap Cluster ODP
    # Kita perlu proyeksi ke meter (misal UTM) untuk perhitungan centroid yang akurat, lalu balik ke WGS84
    # Cara cepat: pakai perkiraan rata-rata lat/lon (cukup untuk area kecil)
    centroids_data = []
    for cluster_id, group in df_points.groupby(cluster_col):
        center_lon = group['lon'].mean()
        center_lat = group['lat'].mean()
        centroids_data.append({'odp_id': cluster_id, 'geometry': Point(center_lon, center_lat)})
    
    gdf_centroids = gpd.GeoDataFrame(centroids_data, crs="EPSG:4326")

    # 3. Buat Masker Pembatas (Area Proyek Total)
    # Ini agar Voronoi tidak meluas sampai tak terhingga. Kita pakai Convex Hull dari SEMUA titik + buffer sedikit.
    total_hull = gdf_points.unary_union.convex_hull
    # Buffer sedikit (misal 0.0005 derajat ~ 50 meter) agar titik terluar tidak pas di garis
    mask_polygon = total_hull.buffer(0.0005)
    gdf_mask = gpd.GeoDataFrame(geometry=[mask_polygon], crs="EPSG:4326")

    # 4. Generate Voronoi Diagram dari Centroid
    # Kita gunakan MultiPoint dari semua centroid sebagai input
    centroids_multipoint = MultiPoint(gdf_centroids['geometry'].tolist())
    # Envelope adalah kotak pembatas untuk perhitungan awal Voronoi
    voronoi_raw = voronoi_diagram(centroids_multipoint, envelope=mask_polygon.envelope)
    
    # Konversi hasil Voronoi mentah menjadi GeoDataFrame
    gdf_voronoi_raw = gpd.GeoDataFrame(geometry=list(voronoi_raw.geoms), crs="EPSG:4326")

    # 5. Proses Spasial Join & Clipping (Bagian paling rumit)
    # Masalah: Voronoi dihasilkan urutan acak, kita harus tahu poligon mana milik ODP ID berapa.
    # Solusi: Lakukan Spatial Join antara poligon Voronoi dengan titik Centroid-nya.
    gdf_voronoi_joined = gpd.sjoin(gdf_voronoi_raw, gdf_centroids, how="inner", predicate="contains")

    # 6. Potong (Clip) Voronoi dengan Masker Area Proyek
    # Agar pinggirannya rapi mengikuti bentuk area proyek secara keseluruhan
    gdf_final_boundaries = gpd.clip(gdf_voronoi_joined, gdf_mask)

    return gdf_final_boundaries, gdf_points

# --- UI UTAMA ---
uploaded_file = st.file_uploader("Unggah File KML/KMZ (Titik Homepass)", type=['kml', 'kmz'])

st.sidebar.header("Pengaturan Jaringan")
max_capacity = st.sidebar.number_input("Kapasitas Maksimal per ODP", min_value=4, max_value=64, value=16)
chunk_size = st.sidebar.number_input("Ukuran Area Makro (Untuk Kecepatan)", min_value=100, max_value=2000, value=500)
opacity_slider = st.sidebar.slider("Transparansi Warna Area", 0.1, 1.0, 0.5)

if uploaded_file is not None:
    with st.spinner('Mengekstrak data spasial...'):
        try:
            df_hp = process_spatial_file(uploaded_file)
            st.success(f"Data dimuat: {len(df_hp)} titik homepass.")
            
            if st.button("Mulai Proses Clustering Rapi"):
                with st.spinner('Langkah 1/3: Menghitung Clustering (Makro & Mikro)...'):
                    # --- PROSES CLUSTERING (Sama seperti sebelumnya untuk kecepatan) ---
                    coords = df_hp[['lat', 'lon']].values
                    n_macro_clusters = max(1, len(df_hp) // chunk_size)
                    kmeans_macro = KMeans(n_clusters=n_macro_clusters, random_state=42, n_init='auto')
                    df_hp['macro_id'] = kmeans_macro.fit_predict(coords)
                    
                    df_hp['final_odp_id'] = -1
                    global_odp_counter = 0
                    
                    for macro_id in range(n_macro_clusters):
                        mask = df_hp['macro_id'] == macro_id
                        macro_data = df_hp[mask]
                        if len(macro_data) == 0: continue
                        
                        n_micro = int(np.ceil(len(macro_data) / max_capacity))
                        # Penanganan jika titik sangat sedikit
                        n_micro = max(1, n_micro)
                        if len(macro_data) < n_micro: n_micro = len(macro_data) # Safety check

                        clf = KMeansConstrained(n_clusters=n_micro, size_min=1, size_max=max_capacity, random_state=42)
                        micro_labels = clf.fit_predict(macro_data[['lat', 'lon']].values)
                        df_hp.loc[mask, 'final_odp_id'] = micro_labels + global_odp_counter
                        global_odp_counter += n_micro
                    
                    num_odp_created = df_hp['final_odp_id'].nunique()
                    st.info(f"Clustering selesai. Terbentuk {num_odp_created} ODP.")

                with st.spinner('Langkah 2/3: Membuat Batas Wilayah Rapi (Voronoi)...'):
                    # --- PROSES GEOMETRI VORONOI BARU ---
                    # Hanya jalankan jika jumlah ODP > 1, Voronoi butuh minimal 2 titik
                    if num_odp_created > 1:
                        gdf_boundaries, gdf_points_all = create_voronoi_boundaries(df_hp)
                    else:
                        # Fallback jika cuma ada 1 ODP, pakai Convex Hull biasa + buffer
                        gdf_points_all = gpd.GeoDataFrame(df_hp, geometry=gpd.points_from_xy(df_hp.lon, df_hp.lat), crs="EPSG:4326")
                        hull = gdf_points_all.unary_union.convex_hull.buffer(0.0002)
                        gdf_boundaries = gpd.GeoDataFrame({'odp_id': [df_hp['final_odp_id'].iloc[0]], 'geometry': [hull]}, crs="EPSG:4326")

                with st.spinner('Langkah 3/3: Merender Peta Interaktif...'):
                    # --- RENDER PETA ---
                    # Siapkan palet warna yang banyak dan berbeda
                    base_colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
                    
                    # Fungsi untuk mendapatkan warna berdasarkan ODP ID
                    def get_color(odp_id):
                        return base_colors[int(odp_id) % len(base_colors)]

                    center_lat, center_lon = df_hp['lat'].mean(), df_hp['lon'].mean()
                    m = folium.Map(location=[center_lat, center_lon], zoom_start=15, prefer_canvas=True, tiles="CartoDB positron")

                    # 1. Layer Area Coverage (Poligon Rapi)
                    folium.GeoJson(
                        gdf_boundaries,
                        name="Area Coverage ODP",
                        style_function=lambda feature: {
                            'fillColor': get_color(feature['properties']['odp_id']),
                            'color': 'black', # Garis pinggir hitam tipis agar tegas
                            'weight': 1,
                            'fillOpacity': opacity_slider
                        },
                        tooltip=folium.GeoJsonTooltip(fields=['odp_id'], aliases=['ODP ID:'])
                    ).add_to(m)

                    # 2. Layer Titik Homepass (Titik kecil)
                    # Kita warnai titiknya sesuai warna areanya agar serasi
                    for _, row in df_hp.iterrows():
                        pt_color = get_color(row['final_odp_id'])
                        folium.CircleMarker(
                            location=[row['lat'], row['lon']],
                            radius=2,
                            color='black',
                            weight=0.5,
                            fill=True,
                            fillColor=pt_color,
                            fillOpacity=1.0,
                            popup=f"HP ID: {row['id']} | ODP: {row['final_odp_id']}"
                        ).add_to(m)

                    folium.LayerControl().add_to(m)
                    st.success("Selesai! Peta telah diperbarui dengan batas yang rapi.")
                    st_folium(m, width=1200, height=700, returned_objects=[])
                    
        except Exception as e:
            st.error(f"Terjadi kesalahan detail: {e}")
            import traceback
            st.code(traceback.format_exc()) # Tampilkan error lengkap untuk debugging
else:
    st.info("Silakan unggah file KML atau KMZ berisi titik homepass.")