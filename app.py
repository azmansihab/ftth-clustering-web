import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, MultiPoint
from shapely.ops import voronoi_diagram
import folium
from streamlit_folium import st_folium
from k_means_constrained import KMeansConstrained
import fiona
import zipfile
import os
import matplotlib.colors as mcolors
import osmnx as ox
import warnings

# Mengabaikan warning bawaan
warnings.filterwarnings('ignore')

fiona.drvsupport.supported_drivers['KML'] = 'rw'
fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'

st.set_page_config(layout="wide", page_title="FTTH ODP Clustering (Rapi & Terarah)")

st.title("🗺️ FTTH Clustering: Sapuan Kanan ke Kiri (Timur ke Barat)")
st.markdown("Menggunakan metode *Spatial Slicing* untuk memastikan area ODP terbentuk lebih konsisten, berurutan dari arah Kanan peta ke Kiri, dibatasi oleh jalan fisik.")

# --- FUNGSI PENDUKUNG ---

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
    west, south, east, north = gdf_points.total_bounds
    buffer_deg = 0.002 
    bbox = (west - buffer_deg, south - buffer_deg, east + buffer_deg, north + buffer_deg)
    
    try:
        roads = ox.features_from_bbox(bbox=bbox, tags={'highway': True})
        return roads
    except Exception as e:
        return None

def create_physical_boundaries(df_points, cluster_col='final_odp_id'):
    gdf_points = gpd.GeoDataFrame(
        df_points, geometry=gpd.points_from_xy(df_points.lon, df_points.lat), crs="EPSG:4326"
    )
    
    centroids_data = []
    for cluster_id, group in df_points.groupby(cluster_col):
        centroids_data.append({'odp_id': cluster_id, 'geometry': Point(group['lon'].mean(), group['lat'].mean())})
    gdf_centroids = gpd.GeoDataFrame(centroids_data, crs="EPSG:4326")

    mask_polygon = gdf_points.unary_union.convex_hull.buffer(0.001)
    gdf_mask = gpd.GeoDataFrame(geometry=[mask_polygon], crs="EPSG:4326")
    
    centroids_multipoint = MultiPoint(gdf_centroids['geometry'].tolist())
    voronoi_raw = voronoi_diagram(centroids_multipoint, envelope=mask_polygon.envelope)
    gdf_voronoi_raw = gpd.GeoDataFrame(geometry=list(voronoi_raw.geoms), crs="EPSG:4326")
    
    gdf_voronoi = gpd.sjoin(gdf_voronoi_raw, gdf_centroids, how="inner", predicate="contains")
    gdf_voronoi = gpd.clip(gdf_voronoi, gdf_mask)

    roads_gdf = get_road_network(gdf_points)
    
    if roads_gdf is not None and not roads_gdf.empty:
        gdf_voronoi_m = gdf_voronoi.to_crs(epsg=3857)
        roads_m = roads_gdf.to_crs(epsg=3857)
        gdf_points_m = gdf_points.to_crs(epsg=3857)
        
        road_buffer = roads_m.geometry.buffer(3) 
        road_union = road_buffer.unary_union
        
        gdf_voronoi_cut = gdf_voronoi_m.copy()
        gdf_voronoi_cut['geometry'] = gdf_voronoi_m.geometry.difference(road_union)
        gdf_voronoi_cut = gdf_voronoi_cut[~gdf_voronoi_cut.geometry.is_empty]
        gdf_exploded = gdf_voronoi_cut.explode(index_parts=False).reset_index(drop=True)
        
        final_polygons = []
        for _, odp_area in gdf_exploded.iterrows():
            poly = odp_area.geometry 
            odp_id = odp_area['odp_id']
            points_in_this_odp = gdf_points_m[gdf_points_m[cluster_col] == odp_id]
            if not points_in_this_odp.empty and points_in_this_odp.geometry.within(poly).any():
                final_polygons.append({'odp_id': odp_id, 'geometry': poly})
                
        if final_polygons:
            gdf_final = gpd.GeoDataFrame(final_polygons, crs="EPSG:3857")
            return gdf_final.to_crs(epsg=4326), gdf_points
            
    return gdf_voronoi, gdf_points

# --- UI UTAMA STREAMLIT ---

st.sidebar.header("Pengaturan Jaringan")
# Default diubah menjadi 20 sesuai permintaan
max_capacity = st.sidebar.number_input("Kapasitas Maks per ODP", min_value=4, max_value=64, value=20)
chunk_size = st.sidebar.number_input("Ukuran Pita Sapuan (Titik)", min_value=50, max_value=1000, value=200)
opacity_slider = st.sidebar.slider("Transparansi Warna", 0.1, 1.0, 0.4)

uploaded_file = st.file_uploader("Unggah File KML/KMZ (Titik Homepass)", type=['kml', 'kmz'])

if uploaded_file is not None:
    with st.spinner('Membaca data file spasial...'):
        df_hp = process_spatial_file(uploaded_file)
        
        if st.button("Mulai Proses Boundary (Kanan ke Kiri)"):
            
            with st.spinner('Langkah 1: Mengurutkan titik dari Kanan ke Kiri & Clustering...'):
                
                # --- LOGIKA BARU: SPATIAL SWEEPING (KANAN KE KIRI) ---
                # 1. Mengurutkan data berdasarkan Longitude (Bujur) dari Timur (Kanan) ke Barat (Kiri)
                df_hp = df_hp.sort_values(by='lon', ascending=False).reset_index(drop=True)
                
                # 2. Membagi titik menjadi "Pita/Irisan Vertikal"
                # Karena data sudah urut Kanan->Kiri, membaginya dengan qcut akan membuat pita-pita dari Timur ke Barat
                n_macro = max(1, len(df_hp) // chunk_size)
                if n_macro > 1:
                    df_hp['macro_id'] = pd.qcut(df_hp.index, q=n_macro, labels=False, duplicates='drop')
                else:
                    df_hp['macro_id'] = 0
                
                df_hp['final_odp_id'] = -1
                global_odp_counter = 0
                
                # 3. Proses Clustering di dalam masing-masing pita vertikal
                for macro_id in sorted(df_hp['macro_id'].unique()):
                    mask = df_hp['macro_id'] == macro_id
                    macro_data = df_hp[mask]
                    if len(macro_data) == 0: continue
                    
                    # Agar tidak acak di dalam pita, kita bisa urutkan lagi berdasarkan Latitude (Utara-Selatan)
                    macro_data = macro_data.sort_values(by='lat', ascending=False)
                    
                    n_micro = max(1, int(np.ceil(len(macro_data) / max_capacity)))
                    clf = KMeansConstrained(n_clusters=n_micro, size_min=1, size_max=max_capacity, random_state=42)
                    
                    # Fit algoritma
                    micro_labels = clf.fit_predict(macro_data[['lat', 'lon']].values)
                    df_hp.loc[macro_data.index, 'final_odp_id'] = micro_labels + global_odp_counter
                    global_odp_counter += n_micro

            with st.spinner('Langkah 2: Memotong batas area dengan jaringan jalan OSM...'):
                gdf_boundaries, gdf_points_all = create_physical_boundaries(df_hp)

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
                
                folium.GeoJson(
                    gdf_boundaries,
                    name="Boundary ODP",
                    style_function=lambda f: {
                        'fillColor': get_color(f['properties']['odp_id']),
                        'color': 'white', 
                        'weight': 1.5,
                        'fillOpacity': opacity_slider
                    },
                    tooltip=folium.GeoJsonTooltip(fields=['odp_id'], aliases=['ODP ID:'])
                ).add_to(m)

                for _, row in df_hp.iterrows():
                    folium.CircleMarker(
                        location=[row['lat'], row['lon']], 
                        radius=3,
                        color='black', weight=1, fill=True, fillColor='#ffcc00', fillOpacity=1,
                        popup=f"ODP: {row['final_odp_id']}"
                    ).add_to(m)

                st.subheader(f"Selesai: {global_odp_counter} ODP Terbentuk (Arah Kanan ke Kiri)")
                st_folium(m, width=1200, height=700, returned_objects=[])

else:
    st.info("Unggah KML/KMZ Anda untuk memulai.")