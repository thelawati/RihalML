import streamlit as st
import pandas as pd
import os
from datetime import datetime
from data_processing import *

# Constants
MODEL_PATH = "a3crime_model.pkl"
DATA_FILE = "crime_reports.csv"
COMPETITION_DATA = "Competition_Dataset.csv"

# Caching model loading
@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)

# Caching competition dataset
@st.cache_data
def get_competition_data():
    if os.path.exists(COMPETITION_DATA):
        df = pd.read_csv(COMPETITION_DATA)
        df.rename(columns={
            'Latitude (Y)': 'Longitude',
            'Longitude (X)': 'Latitude'
        }, inplace=True)
        df['Severity'] = map_severity(df['Category'])
        return df[COLUMN_LIST]
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
        min_date = df_display.loc[valid_dates, 'Dates'].min().date()
        max_date = df_display.loc[valid_dates, 'Dates'].max().date()
        selected_range = st.sidebar.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

        if selected_range and len(selected_range) == 2:
            start, end = selected_range
            df_display = df_display[
                (df_display['Dates'] >= pd.to_datetime(start)) &
                (df_display['Dates'] <= pd.to_datetime(end))
            ]

    # Category
    all_categories = pd.concat([df_pdf, df_comp], ignore_index=True)['Category'].dropna().unique().tolist()
    selected_categories = st.sidebar.multiselect("Category", all_categories)
    if selected_categories:
        df_display = df_display[df_display['Category'].isin(selected_categories)]

    # Severity
    all_severity = pd.concat([df_pdf, df_comp], ignore_index=True)['Severity'].dropna().unique().tolist()
    selected_severity = st.sidebar.multiselect("Severity", all_severity)
    if selected_severity:
        df_display = df_display[df_display['Severity'].isin(selected_severity)]

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

        # Hour
    df_display['Hour'] = df_display['Dates'].dt.hour
    all_hours = sorted(df_display['Hour'].dropna().unique().tolist())
    selected_hours = st.sidebar.multiselect("Hour of Day", all_hours)
    if selected_hours:
        df_display = df_display[df_display['Hour'].isin(selected_hours)]

    # Time of Day
    bins = [0, 6, 12, 18, 24]
    labels = ['Late Night', 'Morning', 'Afternoon', 'Evening']
    df_display['TimeOfDay'] = pd.cut(df_display['Hour'], bins=bins, labels=labels, right=False, include_lowest=True)
    all_times = df_display['TimeOfDay'].dropna().unique().tolist()
    selected_times = st.sidebar.multiselect("Time of Day", all_times)
    if selected_times:
        df_display = df_display[df_display['TimeOfDay'].isin(selected_times)]

    return df_display

# Load model
model = get_model()

# Load existing uploaded PDF data
if os.path.exists(DATA_FILE):
    df_pdf = pd.read_csv(DATA_FILE)
    df_pdf = df_pdf[COLUMN_LIST]
    df_pdf['Dates'] = pd.to_datetime(df_pdf['Dates'], errors='coerce')
else:
    df_pdf = pd.DataFrame(columns=COLUMN_LIST)

# Load competition dataset
df_comp = get_competition_data()
df_comp['Dates'] = pd.to_datetime(df_comp['Dates'], errors='coerce')

# --- Streamlit UI ---
st.title("\U0001F4C2 Police Crime Report Analyzer")

# File uploader
uploaded_files = st.file_uploader("Upload Police Crime Reports (PDF)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        extracted = extract_from_pdf(uploaded_file)
        df_new = standardize_record(extracted, model)
        df_new = df_new.applymap(lambda x: x.upper() if isinstance(x, str) else x)
        df_pdf = pd.concat([df_pdf, df_new], ignore_index=True)

    df_pdf.to_csv(DATA_FILE, index=False)
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
ECHO is on.
