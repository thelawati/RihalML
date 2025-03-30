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
    # Combine datasets based on choice
    if dataset_choice == "All Data":
        df_display = pd.concat([df_pdf, df_comp], ignore_index=True)
    elif dataset_choice == "PDF Data Only":
        df_display = df_pdf
    else:
        df_display = df_comp

    # Ensure consistent data types and preprocessing
    df_display['Dates'] = pd.to_datetime(df_display['Dates'], errors='coerce')
    df_display = df_display.dropna(subset=['Dates'])  # Remove rows with invalid dates

    # Create sidebar for filtering
    st.sidebar.header("\U0001F50D Advanced Filters")

    # Date Range Filter with Enhanced UI
    min_date = df_display['Dates'].min().date()
    max_date = df_display['Dates'].max().date()
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", min_value=min_date, max_value=max_date, value=min_date)
    with col2:
        end_date = st.date_input("End Date", min_value=min_date, max_value=max_date, value=max_date)

    # Apply date filter
    df_display = df_display[
        (df_display['Dates'].dt.date >= start_date) & 
        (df_display['Dates'].dt.date <= end_date)
    ]

    # Enhanced Categorical Filters with Search
    def create_searchable_multiselect(label, options):
        search_term = st.sidebar.text_input(f"Search {label}", key=f"search_{label}")
        filtered_options = [opt for opt in options if search_term.lower() in str(opt).lower()]
        return st.sidebar.multiselect(label, filtered_options)

    # Categorical Filters with Search
    filters = {
        "Category": df_display['Category'].dropna().unique().tolist(),
        "PdDistrict": df_display['PdDistrict'].dropna().unique().tolist(),
        "Resolution": df_display['Resolution'].dropna().unique().tolist()
    }

    # Apply categorical filters
    for column, options in filters.items():
        selected = create_searchable_multiselect(column, options)
        if selected:
            df_display = df_display[df_display[column].isin(selected)]

    # Time-based Filters
    df_display['Hour'] = df_display['Dates'].dt.hour
    df_display['DayOfWeek'] = df_display['Dates'].dt.day_name()

    # Time of Day Categorization
    def categorize_time(hour):
        if 0 <= hour < 6:
            return 'Late Night'
        elif 6 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 18:
            return 'Afternoon'
        else:
            return 'Evening'

    df_display['Time of Day'] = df_display['Hour'].apply(categorize_time)

    # Additional Time Filters
    time_filters = {
        "Day of Week": df_display['DayOfWeek'].unique().tolist(),
        "Time of Day": df_display['Time of Day'].unique().tolist()
    }

    for column, options in time_filters.items():
        selected = st.sidebar.multiselect(column, options)
        if selected:
            df_display = df_display[df_display[column].isin(selected)]

    # Severity Filter with Slider
    severity_options = sorted(df_display['Severity'].dropna().unique().tolist())
    if severity_options:
        if len(set(severity_options)) > 1:
            severity_range = st.sidebar.slider(
                "Severity Range", 
                min_value=min(severity_options), 
                max_value=max(severity_options), 
                value=(min(severity_options), max(severity_options))
            )
            df_display = df_display[
                (df_display['Severity'] >= severity_range[0]) & 
                (df_display['Severity'] <= severity_range[1])
            ]
        else:
            # Checkbox to filter or show all when single severity
            show_all = st.sidebar.checkbox(f"Show All (Severity: {severity_options[0]})", value=True)
            if not show_all:
                df_display = df_display[df_display['Severity'] == severity_options[0]]
    # Statistics and Insights
    st.sidebar.markdown("### Dataset Insights")
    st.sidebar.write(f"Total Records: {len(df_display)}")
    
    # Pie chart of categories
    if not df_display.empty:
        category_counts = df_display['Category'].value_counts()
        st.sidebar.markdown("#### Category Distribution")
        st.sidebar.bar_chart(category_counts)

    # Debugging and Transparency
    st.sidebar.markdown("---")
    st.sidebar.write(f"📁 View: {dataset_choice}")
    st.sidebar.write(f"📊 Rows after filtering: {len(df_display)}")

    return df_display
# Load model
model = get_model()

# Load existing uploaded PDF data from GCS
def load_csv_from_gcs(bucket_name, blob_name) -> pd.DataFrame:
    """Load CSV data from Google Cloud Storage with error handling."""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        if not blob.exists():
            st.warning(f"The blob {blob_name} does not exist in bucket {bucket_name}.")
            return pd.DataFrame(columns=COLUMN_LIST)
        
        data = blob.download_as_bytes()
        df = pd.read_csv(BytesIO(data))
        return df.applymap(lambda x: x.upper() if isinstance(x, str) else x)
    
    except Exception as e:
        st.error(f"Error loading CSV from GCS: {str(e)}")
        return pd.DataFrame(columns=COLUMN_LIST)

def save_csv_to_gcs(df: pd.DataFrame, bucket_name, blob_name):
    """Save DataFrame to CSV in Google Cloud Storage with error handling."""
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
    st.write(new_entries)
    
    for uploaded_file in uploaded_files:
        extracted = extract_from_pdf(uploaded_file)
        df_new = standardize_record(extracted, model)
        st.write(df_new)
        df_new = df_new.applymap(lambda x: x.upper() if isinstance(x, str) else x)
        st.write(df_new)
        st.write("pdf before:", df_pdf)
        df_pdf = pd.concat([df_pdf, df_new], ignore_index=True)
        st.write("pdf after:", df_pdf)

    save_csv_to_gcs(df_new, BUCKET_NAME, CSV_FILENAME)
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
