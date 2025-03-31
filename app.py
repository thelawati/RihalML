import streamlit as st
import pandas as pd
from io import BytesIO, StringIO
from google.cloud import storage
import os
from datetime import datetime
from streamlit_folium import folium_static
import folium
from folium.plugins import MarkerCluster
import altair as alt
from data_processing import *
from utils import *

# Load model
model = get_model()

# Load existing pdf data
df_pdf = load_csv_from_gcs(BUCKET_NAME, CSV_FILENAME)

# Load competition dataset
df_comp = get_competition_data()


# --- Streamlit UI ---
st.title("\U0001F4C2 CityX Crime Report Interface")



# File uploader logic
uploaded_files = st.file_uploader("Upload Police Crime Reports (PDF)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    new_entries = []
    
    df_new = pd.DataFrame()
    for uploaded_file in uploaded_files:
        extracted = extract_from_pdf(uploaded_file)
        if df_new.empty:
            df_new = standardize_pdf_record(extracted, model)

        else:
            df_new = pd.concat([df_new, standardize_pdf_record(extracted, model)], ignore_index=True)
        
    df_new['Dates'] = pd.to_datetime(df_new['Dates'], errors='coerce')
    df_new = df_new.applymap(lambda x: x.upper() if isinstance(x, str) else x)

    st.write(df_new)
    df_pdf = pd.concat([df_pdf, df_new], ignore_index=True)
    

    save_csv_to_gcs(df_new, BUCKET_NAME, CSV_FILENAME)
    st.success("\u2705 Reports uploaded and data saved! ")


# Load existing pdf data
df_pdf = load_csv_from_gcs(BUCKET_NAME, CSV_FILENAME)
df_pdf = df_pdf.drop_duplicates().reset_index(drop=True)

# Get filtered dataset
df_display = get_display_data(df_pdf, df_comp)


# Crime map view
st.markdown("---")
st.subheader("🗺 Crime Map & Table View")
st.write("Map and Table will show a maximum of 1000 crimes. Utilize the filters to drill in!")

# Prepare map data
df_map = df_display.dropna(subset=['Latitude', 'Longitude']).copy()[:1000]

try:
    df_map['Latitude'] = df_map['Latitude'].astype(float)
    df_map['Longitude'] = df_map['Longitude'].astype(float)

    if not df_map.empty:
        center_lat = df_map['Latitude'].mean()
        center_lon = df_map['Longitude'].mean()

        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
        marker_cluster = MarkerCluster().add_to(m)

        for _, row in df_map.iterrows():
            popup_info = f"<b>Category:</b> {row['Category']}<br><b>Description:</b> {row['Descript']}<br><b>Date:</b> {row['Dates']}"
            folium.Marker(
                location=[row['Latitude'], row['Longitude']],
                popup=popup_info
            ).add_to(marker_cluster)

        folium_static(m)
    else:
        st.info("No location data available to display the map.")
except Exception as e:
    st.error(f"Failed to render map: {e}")



# Showcase top 1000 records only -- df_map is the first 1000 rows of df_display
st.dataframe(df_map) 




st.markdown("---")
st.subheader("🚨 Crime Overview Snapshot")
display_crime_snapshot(df_display)