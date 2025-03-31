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




# Constants
COMPETITION_DATA = "Competition_Dataset.csv"
MODEL_PATH = "a3crime_model.pkl"
BUCKET_NAME = "rihal-ml-storage-001"
CSV_FILENAME = "crime_reports.csv"
DATA_FILE = "crime_reports.csv"


# Caching model loading
@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


# Caching competition dataset
@st.cache_data()
def get_competition_data():
    #Connect to Google Storage
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(COMPETITION_DATA)
    if blob.exists():
        data = blob.download_as_bytes()
        df = pd.read_csv(BytesIO(data))

        #Process competition data using function from data_processing
        return process_comp_data(df)
    #Return empty dataframe otherwise
    return pd.DataFrame(columns=COLUMN_LIST)


# Load existing uploaded PDF data from GCS
def load_csv_from_gcs(bucket_name, blob_name) -> pd.DataFrame:
    """Load CSV data from Google Cloud Storage with error handling."""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        if blob.exists():
            data = blob.download_as_bytes()
            df = pd.read_csv(BytesIO(data))

            if 'Dates' in df.columns:
                df['Dates'] = df['Dates'].astype(str).str.strip()
                df['Dates'] = pd.to_datetime(df['Dates'], errors='coerce')
            
        else:
            return pd.DataFrame(columns=COLUMN_LIST)
        
        return df.applymap(lambda x: x.upper() if isinstance(x, str) else x)
    
    except Exception as e:
        
        return pd.DataFrame(columns=COLUMN_LIST)
    

def save_csv_to_gcs(df: pd.DataFrame, bucket_name, blob_name):
    
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        # Check if existing data is present and skip duplicating
        if blob.exists():
            existing_data = load_csv_from_gcs(bucket_name, blob_name)
            df = pd.concat([existing_data, df]).drop_duplicates().reset_index(drop=True)
        
        with StringIO() as csv_buffer:
            df.to_csv(csv_buffer, index=False)
            blob.upload_from_string(csv_buffer.getvalue(), content_type="text/csv")
        st.success("Data saved to GCS successfully!")
    
    except Exception as e:
        st.error(f"Error saving CSV to GCS: {str(e)}")




# UI filters sidebar build -- This function builds the filters and displays the options based on the dataframe input
def display_filter_sidebar(df_display):
    dataset_choice = st.sidebar.radio(
        "Select Data to Display:",
        ("All Data", "PDF Data Only", "Competition Data Only"),
         key="dataset_choice_radio" 
    )

    st.sidebar.header("\U0001F50D Advanced Filters")
    filters = {}

    min_date = df_display['Dates'].min().date()
    max_date = df_display['Dates'].max().date()

    col1, col2 = st.sidebar.columns(2)
    with col1:
        filters['start_date'] = st.date_input("Start Date", min_value=min_date, max_value=max_date, value=min_date)
    with col2:
        filters['end_date'] = st.date_input("End Date", min_value=min_date, max_value=max_date, value=max_date)

    filters['Category'] = st.sidebar.multiselect("Category", df_display['Category'].dropna().unique().tolist())
    filters['PdDistrict'] = st.sidebar.multiselect("PdDistrict", df_display['PdDistrict'].dropna().unique().tolist())
    filters['Resolution'] = st.sidebar.multiselect("Resolution", df_display['Resolution'].dropna().unique().tolist())

    filters['DayOfWeek'] = st.sidebar.multiselect("Day of Week", df_display['DayOfWeek'].unique().tolist())
    filters['Time of Day'] = st.sidebar.multiselect("Time of Day", df_display['Time of Day'].unique().tolist())

    severity_options = sorted(df_display['Severity'].dropna().unique().tolist())
    if severity_options:
        if len(set(severity_options)) > 1:
            filters['Severity Range'] = st.sidebar.slider(
                "Severity Range",
                min_value=min(severity_options),
                max_value=max(severity_options),
                value=(min(severity_options), max(severity_options))
            )
        else:
            show_all = st.sidebar.checkbox(f"Show All (Severity: {severity_options[0]})", value=True)
            filters['Severity Single'] = severity_options[0] if not show_all else None

    return dataset_choice, filters

# Apply filtering logic -- this function takes the dataframe displaying data, and the filters the user creates to display the relevant data
def apply_filters(df, filters):
    df = df[
        (df['Dates'].dt.date >= filters['start_date']) &
        (df['Dates'].dt.date <= filters['end_date'])
    ]

    for col in ['Category', 'PdDistrict', 'Resolution']:
        if filters[col]:
            df = df[df[col].isin(filters[col])]

    if filters['DayOfWeek']:
        df = df[df['DayOfWeek'].isin(filters['DayOfWeek'])]

    if filters['Time of Day']:
        df = df[df['Time of Day'].isin(filters['Time of Day'])]

    if 'Severity Range' in filters:
        df = df[
            (df['Severity'] >= filters['Severity Range'][0]) &
            (df['Severity'] <= filters['Severity Range'][1])
        ]
    elif filters.get('Severity Single') is not None:
        df = df[df['Severity'] == filters['Severity Single']]

    return df


def get_display_data(df_pdf, df_comp):
    # Combine all data for filtering sidebar setup
    df_all = pd.concat([df_pdf, df_comp], ignore_index=True)
    df_all['Dates'] = pd.to_datetime(df_all['Dates'], errors='coerce')
    df_all = df_all.dropna(subset=['Dates'])
    df_all['Hour'] = df_all['Dates'].dt.hour
    df_all['DayOfWeek'] = df_all['Dates'].dt.day_name()

    def categorize_time(hour):
        if 0 <= hour < 6:
            return 'Late Night'
        elif 6 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 18:
            return 'Afternoon'
        else:
            return 'Evening'

    df_all['Time of Day'] = df_all['Hour'].apply(categorize_time)

    dataset_choice, filters = display_filter_sidebar(df_all)

    if dataset_choice == "All Data":
        df_display = df_all.copy()
    elif dataset_choice == "PDF Data Only":
        df_display = df_pdf.copy()
    else:
        df_display = df_comp.copy()

    df_display['Dates'] = pd.to_datetime(df_display['Dates'], errors='coerce')
    df_display = df_display.dropna(subset=['Dates'])
    df_display['Hour'] = df_display['Dates'].dt.hour
    df_display['DayOfWeek'] = df_display['Dates'].dt.day_name()
    df_display['Time of Day'] = df_display['Hour'].apply(categorize_time)

    df_filtered = apply_filters(df_display, filters)

    st.sidebar.markdown("### Dataset Insights")
    st.sidebar.write(f"Total Records: {len(df_filtered)}")

    if not df_filtered.empty:
        category_counts = df_filtered['Category'].value_counts()
        st.sidebar.markdown("#### Category Distribution")
        st.sidebar.bar_chart(category_counts)

    st.sidebar.markdown("---")
    st.sidebar.write(f"📁 View: {dataset_choice}")
    st.sidebar.write(f"📊 Rows after filtering: {len(df_filtered)}")

    return df_filtered

def display_crime_snapshot(df):


    # Ensure Dates column is datetime
    df['Dates'] = pd.to_datetime(df['Dates'])

    # 2x2 Grid layout
    cols1 = st.columns(2)
    cols2 = st.columns(2)

    # Plot 1: Crime Count by Category
    with cols1[0]:
        st.markdown("**Top 5 Crime Categories**")
        top_categories = df['Category'].value_counts().head(5).reset_index()
        top_categories.columns = ['Category', 'Count']
        chart = alt.Chart(top_categories).mark_bar().encode(
            x=alt.X('Category:N', sort='-y'),
            y='Count:Q',
            tooltip=['Category', 'Count']
        ).properties(height=250)
        st.altair_chart(chart, use_container_width=True)

    # Plot 2: Monthly Crime Trend
    with cols1[1]:
        st.markdown("**Monthly Crime Trend**")
        monthly = df.groupby(df['Dates'].dt.to_period('M')).size().reset_index(name='Count')
        monthly['Dates'] = monthly['Dates'].dt.to_timestamp()
        #Dropping the last month as it wasnt a full month of crime
        monthly = monthly.iloc[:-1] 
        chart = alt.Chart(monthly).mark_line(point=True).encode(
            x='Dates:T',
            y='Count:Q',
            tooltip=['Dates', 'Count']
        ).properties(height=250)
        st.altair_chart(chart, use_container_width=True)

    # Plot 3: Top Police Districts
    with cols2[0]:
        st.markdown("**Top 5 Police Districts**")
        top_districts = df['PdDistrict'].value_counts().head(5).reset_index()
        top_districts.columns = ['PdDistrict', 'Count']
        chart = alt.Chart(top_districts).mark_bar().encode(
            x='Count:Q',
            y=alt.Y('PdDistrict:N', sort='-x'),
            tooltip=['PdDistrict', 'Count']
        ).properties(height=250)
        st.altair_chart(chart, use_container_width=True)

    # Plot 4: Severity Distribution
    with cols2[1]:
        st.markdown("**Severity Distribution**")
        if df['Severity'].nunique() > 10:
            st.warning("Too many unique severity values to plot clearly.")
        else:
            severity_dist = df['Severity'].value_counts().reset_index()
            severity_dist.columns = ['Severity', 'Count']
            chart = alt.Chart(severity_dist).mark_bar().encode(
                x='Severity:N',
                y='Count:Q',
                tooltip=['Severity', 'Count']
            ).properties(height=250)
            st.altair_chart(chart, use_container_width=True)

