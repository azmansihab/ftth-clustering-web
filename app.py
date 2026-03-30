import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, MultiPoint
from shapely.ops import voronoi_diagram
import folium
from streamlit_folium import st_folium
from k_means_constrained import KMeansConstrained
from sklearn.cluster import AgglomerativeClustering
import fiona
import zipfile
import os
import matplotlib.colors as mcolors
import osmnx as ox
import simplekml
import warnings

# Mengabaikan warning bawaan
warnings.filterwarnings('ignore')

fiona.drvsupport.supported_drivers['KML'] = 'rw'
fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'

st.set_page_config(layout="wide", page_title="FTTH ODP Clustering (Distance Based)")

st.title("🗺️ FTTH Clustering: Organik Berdasarkan Jarak (Maks 200m)")
st.markdown("Mengelompokkan titik secara natural/mencar ke titik terdekat dalam batas jarak maksimal (Meter), dipadukan dengan batas maksimal kapasitas ODP (16 port), lalu dibungkus dengan Voronoi yang dipotong jalan.")

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
    buffer_deg = 0.003
    bbox = (west - buffer_deg, south - buffer_deg, east + buffer_deg, north + buffer_deg)
    try:
        roads = ox.features_from_bbox(bbox=bbox, tags={'highway': True})
        return roads
    except Exception as e:
        return None

def create_voronoi_road_boundaries(df_points, cluster_col='final_odp_id'):
    gdf_points = gpd.GeoDataFrame(
        df_points, geometry=gpd.points_from_xy(df_points.lon, df_points.lat), crs="EPSG:4326"
    )
    
    centroids_data = []
    for cluster_id, group in df_points.groupby(cluster_col):
        centroids_data.append({'odp_id': cluster_id, 'geometry': Point(group['lon'].mean(), group['lat'].mean())})
    gdf_centroids = gpd.GeoDataFrame(centroids_data, crs="EPSG:4326")

    mask_polygon = gdf_points.unary_union.convex_hull.buffer(0.0015)
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

def export_colored_kml(gdf_boundaries, filepath, get_color_func, opacity_float):
    kml = simplekml.Kml()
    def to_kml_color(hex_str, alpha):
        hex_str = hex_str.lstrip('#')
        if len(hex_str) == 3: hex_str = ''.join([c*2 for c in hex_str])
        r, g, b = hex_str[0:2], hex_str[2:4], hex_str[4:6]
        a = f"{int(alpha * 255):02X}"
        return f"{a}{b}{g}{r}"

    for _, row in gdf_boundaries.iterrows():
        odp_id = row['odp_id']
        geom = row['geometry']
        hex_color = get_color_func(odp_id)
        kml_color = to_kml_color(hex_color, opacity_float)
        
        geoms = [geom] if geom.geom_type == 'Polygon' else geom.geoms
        
        for g in geoms:
            pol = kml.newpolygon(name=f"ODP {odp_id}")
            pol.outerboundaryis = list(g.exterior.coords)
            pol.innerboundaryis = [list(i.coords) for i in g.interiors]
            pol.style.polystyle.color = kml_color
            pol.style.linestyle.color = to_kml_color('#ffffff', 1.0)
            pol.style.linestyle.width = 2
    kml.save(filepath)

# --- UI UTAMA STREAMLIT ---

st.sidebar.header("Pengaturan Jaringan")
max_capacity = st.sidebar.number_input("Kapasitas Maks per ODP", min_value=4, max_value=64, value=16)

# PENGATURAN BARU: JARAK MAKSIMAL DALAM METER
max_distance = st.sidebar.number_input("Jarak Maksimal antar Rumah (Meter)", min_value=50, max_value=1000, value=200)

opacity_slider = st.sidebar.slider("Transparansi Warna Area", 0.1, 1.0, 0.45)

uploaded_file = st.file_uploader("Unggah File KML/KMZ (Titik Homepass)", type=['kml', 'kmz'])

if uploaded_file is not None:
    with st.spinner('Membaca data file spasial...'):
        df_hp = process_spatial_file(uploaded_file)
        
        if st.button("Mulai Proses (Organik Berdasarkan Jarak)"):
            
            base_colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
            def get_color(odp_id): return base_colors[int(odp_id) % len(base_colors)]

            with st.spinner(f'Langkah 1: Clustering Jarak {max_distance}m & Kapasitas {max_capacity}...'):
                
                # 1. Konversi Koordinat ke Meter (EPSG:3857) untuk mengukur jarak akurat
                gdf_m = gpd.GeoDataFrame(
                    df_hp, geometry=gpd.points_from_xy(df_hp.lon, df_hp.lat), crs="EPSG:4326"
                ).to_crs(epsg=3857)
                
                coords_m = np.column_stack((gdf_m.geometry.x, gdf_m.geometry.y))
                
                # 2. Agglomerative Clustering (Mengelompokkan rumah yang jaraknya <= 200m)
                agg_cluster = AgglomerativeClustering(
                    n_clusters=None, 
                    distance_threshold=max_distance, 
                    linkage='complete' # Complete linkage memastikan diameter area maksimal adalah distance_threshold
                )
                df_hp['dist_cluster'] = agg_cluster.fit_predict(coords_m)
                
                # 3. Pengecekan Kapasitas (Maks 16 per area jarak)
                df_hp['final_odp_id'] = -1
                global_odp_counter = 0
                
                for dist_c in df_hp['dist_cluster'].unique():
                    mask = df_hp['dist_cluster'] == dist_c
                    group_data = df_hp[mask]
                    
                    if len(group_data) > max_capacity:
                        # Jika area tersebut ternyata isinya lebih dari 16 rumah, kita potong lagi
                        n_micro = int(np.ceil(len(group_data) / max_capacity))
                        clf = KMeansConstrained(n_clusters=n_micro, size_min=1, size_max=max_capacity, random_state=42)
                        micro_labels = clf.fit_predict(group_data[['lat', 'lon']].values)
                        df_hp.loc[mask, 'final_odp_id'] = micro_labels + global_odp_counter
                        global_odp_counter += n_micro
                    else:
                        # Jika sudah aman di bawah 16, langsung jadikan 1 ODP
                        df_hp.loc[mask, 'final_odp_id'] = global_odp_counter
                        global_odp_counter += 1

            with st.spinner('Langkah 2: Menarik data Jalan OSM & Membentuk Voronoi Rapat (Mohon tunggu)...'):
                gdf_boundaries, gdf_points_all = create_voronoi_road_boundaries(df_hp)

            with st.spinner('Langkah 3: Merender Peta Satelit...'):
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
                        radius=2.5,
                        color='black', weight=1, fill=True, fillColor='#ffcc00', fillOpacity=1,
                        popup=f"ODP: {row['final_odp_id']}"
                    ).add_to(m)

                st.subheader(f"Selesai: {global_odp_counter} ODP Terbentuk secara Organik")
                
                # --- FITUR DOWNLOAD WARNA ---
                st.markdown("### 📥 Unduh Hasil Desain (Export)")
                col1, col2 = st.columns(2)
                
                with col1:
                    csv_data = df_hp.to_csv(index=False).encode('utf-8')
                    st.download_button("📄 Unduh Data Homepass (CSV)", data=csv_data, file_name="hasil_clustering_homepass.csv", mime="text/csv")
                    
                with col2:
                    kml_file_path = "hasil_boundary_organik_berwarna.kml"
                    if os.path.exists(kml_file_path):
                        os.remove(kml_file_path)
                        
                    gdf_boundaries_export = gdf_boundaries.to_crs(epsg=4326)
                    export_colored_kml(gdf_boundaries_export, kml_file_path, get_color, opacity_slider)
                    
                    with open(kml_file_path, "rb") as kml_file:
                        st.download_button("🗺️ Unduh Boundary BERWARNA (KML)", data=kml_file, file_name="hasil_boundary_organik_berwarna.kml", mime="application/vnd.google-earth.kml+xml")

                st_folium(m, width=1200, height=700, returned_objects=[])

else:
    st.info("Unggah KML/KMZ Anda untuk memulai.")