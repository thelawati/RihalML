import streamlit as st
import pandas as pd
from io import BytesIO, StringIO
from google.cloud import storage
import os
from datetime import datetime
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
@st.cache_data
def get_competition_data():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(COMPETITION_DATA)
    if blob.exists():
        data = blob.download_as_bytes()
        df = pd.read_csv(BytesIO(data))
        df.rename(columns={
            'Latitude (Y)': 'Longitude',
            'Longitude (X)': 'Latitude'
        }, inplace=True)
        df['Severity'] = map_severity(df['Category'])
        df = df[COLUMN_LIST]
        df = df.applymap(lambda x: x.upper() if isinstance(x, str) else x)
        df['Dates'] = pd.to_datetime(df['Dates'], errors='coerce')
        return df
    return pd.DataFrame(columns=COLUMN_LIST)
    return pd.DataFrame(columns=COLUMN_LIST)

# Function to get selected and filtered data
def get_display_data(dataset_choice, df_pdf, df_comp):
    if dataset_choice == "All Data":
        df_display = pd.concat([df_pdf, df_comp], ignore_index=True)
    elif dataset_choice == "PDF Data Only":
        df_display = df_pdf
    else:
        df_display = df_comp

    # Apply sidebar filters here
    st.sidebar.header("\U0001F50D Filter Options")

    # Dates
    df_display['Dates'] = pd.to_datetime(df_display['Dates'], errors='coerce')
    valid_dates = df_display['Dates'].notna()

    if valid_dates.any():
        df_display = df_display[valid_dates]

        min_date = df_display['Dates'].min().date()
        max_date = df_display['Dates'].max().date()

        selected_range = st.sidebar.date_input(
            "Date Range",
            value=[min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )

        if selected_range and len(selected_range) == 2:
            start, end = selected_range
            df_display = df_display[
                (df_display['Dates'].dt.date >= start) &
                (df_display['Dates'].dt.date <= end)
            ]

    # Category
    all_categories = pd.concat([df_pdf, df_comp], ignore_index=True)['Category'].dropna().unique().tolist()
    selected_categories = st.sidebar.multiselect("Category", all_categories)
    if selected_categories:
        df_display = df_display[df_display['Category'].isin(selected_categories)]

    # Severity
    all_severity = sorted(pd.concat([df_pdf, df_comp], ignore_index=True)['Severity'].dropna().unique().tolist(), key=int)
    selected_severity = st.sidebar.multiselect("SEVERITY", all_severity)
    if selected_severity:
        df_display = df_display[df_display['Severity'].isin(selected_severity)]

    # Hour
    df_display['Hour'] = df_display['Dates'].dt.hour
    all_hours = sorted(df_display['Hour'].dropna().unique().tolist())
    selected_hours = st.sidebar.multiselect("Hour", all_hours)
    if selected_hours:
        df_display = df_display[df_display['Hour'].isin(selected_hours)]

    # Time of Day
    bins = [0, 6, 12, 18, 24]
    labels = ['LATE NIGHT', 'MORNING', 'AFTERNOON', 'EVENING']
    df_display['Time of Day'] = pd.cut(df_display['Hour'], bins=bins, labels=labels, right=False, include_lowest=True)
    all_times = df_display['Time of Day'].dropna().unique().tolist()
    selected_times = st.sidebar.multiselect("Time of Day", all_times)
    if selected_times:
        df_display = df_display[df_display['Time of Day'].isin(selected_times)]

    # Police District
    all_districts = pd.concat([df_pdf, df_comp], ignore_index=True)['PdDistrict'].dropna().unique().tolist()
    selected_districts = st.sidebar.multiselect("Police District", all_districts)
    if selected_districts:
        df_display = df_display[df_display['PdDistrict'].isin(selected_districts)]

    # Resolution
    all_resolutions = pd.concat([df_pdf, df_comp], ignore_index=True)['Resolution'].dropna().unique().tolist()
    selected_resolutions = st.sidebar.multiselect("Resolution", all_resolutions)
    if selected_resolutions:
        df_display = df_display[df_display['Resolution'].isin(selected_resolutions)]

    # Debug info
    st.write("📁 View:", dataset_choice)
    st.write("📊 Rows before filtering:", df_display.shape[0])
    st.write("📤 Rows after filtering:", df_display.shape[0])
    st.write("📅 Dates (Parsed):", df_display['Dates'].dropna().dt.strftime("%Y-%m-%d %H:%M:%S").unique())
    st.write("⏰ Hours:", df_display['Hour'].dropna().unique())
    if 'Time of Day' in df_display.columns and not df_display['Time of Day'].dropna().empty:
        st.write("🕒 Time of Day:", df_display['Time of Day'].dropna().astype(str).unique())
    return df_display

# Load model
model = get_model()

# Load existing uploaded PDF data from GCS
def load_csv_from_gcs(bucket_name, blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    if blob.exists():
        data = blob.download_as_bytes()
        return pd.read_csv(BytesIO(data))
    return pd.DataFrame(columns=COLUMN_LIST)

def save_csv_to_gcs(df, bucket_name, blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    blob.upload_from_string(csv_buffer.getvalue(), content_type="text/csv")

df_pdf = load_csv_from_gcs(BUCKET_NAME, CSV_FILENAME)
df_pdf = df_pdf.applymap(lambda x: x.upper() if isinstance(x, str) else x)

if 'Dates' in df_pdf.columns:
    df_pdf['Dates'] = df_pdf['Dates'].astype(str).str.strip()
    df_pdf['Dates'] = pd.to_datetime(df_pdf['Dates'], errors='coerce')

# Load competition dataset
df_comp = get_competition_data()

# --- Streamlit UI ---
st.title("\U0001F4C2 Police Crime Report Analyzer")

# File uploader
uploaded_files = st.file_uploader("Upload Police Crime Reports (PDF)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    new_entries = []
    for uploaded_file in uploaded_files:
        extracted = extract_from_pdf(uploaded_file)
        df_new = standardize_record(extracted, model)
        df_new = df_new.applymap(lambda x: x.upper() if isinstance(x, str) else x)
        df_pdf = pd.concat([df_pdf, df_new], ignore_index=True)

    save_csv_to_gcs(df_pdf, BUCKET_NAME, CSV_FILENAME)
    st.success("\u2705 Reports uploaded and data saved!")


# Dataset toggle
dataset_choice = st.radio(
    "Select Data to Display:",
    ("All Data", "PDF Data Only", "Competition Data Only")
)

# Get filtered dataset
df_display = get_display_data(dataset_choice, df_pdf, df_comp)

# Display selected data
st.dataframe(df_display)

# Placeholder for future map rendering
st.markdown("---")
st.subheader("\U0001F5FA Crime Map View (Coming Soon)")
