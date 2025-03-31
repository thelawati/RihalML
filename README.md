Markdown

# 2025 ML Rihal Codestacker Challenge: CityX Crime Watch - Operation Safe Streets

## Overview

This project is a Streamlit dashboard designed to visualize and analyze crime data for CityX. It allows users to:

* View crime data on an interactive map using Folium.
* Explore crime statistics through various data visualizations (Altair charts).
* Upload new crime reports (PDFs) and automatically categorize them using a machine learning model.
* Manage and interact with the data stored on Google Cloud Storage.

## Tech Stack

* **Streamlit:** For the interactive web dashboard.
* **Pandas:** For data manipulation and analysis.
* **Folium:** For map visualization.
* **Altair:** For data visualization.
* **Scikit-learn:** For the machine learning model.
* **Google Cloud Storage:** For data storage.
* **Docker:** For containerization.
* **Python:** 3.9
* **Machine Learning Model:**
    * `a3crime_model.pkl`: A Pipeline with `TfidfVectorizer` (ngram_range=(1, 2)) and `LogisticRegression`.

## Files

* `app.py`: Streamlit front-end application.
* `utils.py`: Utility functions for front-end configurations.
* `data_processing.py`: Module for data preprocessing.
* `Dockerfile`: Docker configuration file.
* `Competition_Dataset.csv`: Training dataset for the machine learning model (stored in Google Cloud Storage).
* `crime_reports.csv`: directory where uploaded crime reports are stored.

## Google Cloud Setup

1.  **Google Cloud Project:** Ensure you have a Google Cloud project with the necessary permissions.
2.  **Google Cloud Storage Bucket:**
    * Create a bucket named `rihal-ml-storage-001`.
    * Upload `Competition_Dataset.csv` to this bucket.
    * Ensure the service account running the Cloud Run container has storage object viewer and storage object creator permissions.
3.  **Authentication:**
    * Configure Google Cloud authentication using the Google Cloud CLI or by setting up service account credentials.
4.  **Container Deployment:**
    * Deploy the docker image to Cloud Run.

## Installation/Setup

1.  **Clone the Repository:**
    ```bash
    git clone [repository_url]
    cd [repository_directory]
    ```

2.  **Google Cloud Run Deployment (Recommended):**
    * **Docker Image Build (GCP):**
        * Google Cloud Run will use the provided `Dockerfile` to build the Docker image. This Dockerfile specifies the application's environment and dependencies.
        * The `Dockerfile` performs the following actions:
            * Uses the `python:3.9-slim-buster` base image for a lightweight environment.
            * Sets the working directory to `/app`.
            * Copies all files from the current directory to the container.
            * Installs the Python dependencies from `requirements.txt` using `pip install --no-cache-dir`.
            * Exposes port 8501 for the Streamlit application.
        * You do not need to build the docker image locally. Cloud run will build it.
    * **Google Cloud Run Deployment:**
        * Ensure you have a Google Cloud project and have enabled the Cloud Run API.
        * Use the Google Cloud CLI or the Google Cloud Console to deploy the Docker image to Cloud Run.
        * Cloud Run will automatically handle container orchestration and scaling.
    * **Google Cloud Storage Setup:**
        * Ensure you have a Google Cloud Storage bucket named `rihal-ml-storage-001`.
        * Upload the `Competition_Dataset.csv` file to this bucket.
        * Ensure that the service account being used by the Cloud Run container has the necessary permissions to read and write to this bucket.
    * **Google Cloud CLI (Optional):**
        * If using the command line for Cloud Run deployment, ensure the Google Cloud CLI is configured.

3.  **Local Python Setup (Alternative - for testing only):**
    * If you want to test the application locally (outside of GCP), you can install the dependencies directly:
        ```bash
        pip install -r requirements.txt
        ```
    * Then, run the Streamlit application:
        ```bash
        streamlit run app.py
        ```
        Ensure you have Python 3.9 installed. This is not the recommended deployment method.

## Usage

1.  **Running the Application:**
    * Access the application via the Cloud Run URL provided by GCP.
2.  **Dashboard Features:**
    * **Map:** View crime locations on the interactive map.
    * **Data Visualizations:** Explore crime statistics through charts.
    * **Upload Crime Reports:** Upload PDF crime reports to the application.
    * **Crime Category Prediction:** The application will use the ML model to predict the crime category.
3.  **Uploading Crime Reports:**
    * Use the file upload feature to upload PDF crime reports.
    * The application will display the predicted crime category.
4.  **Interacting with the Map:**
    * Zoom in and out of the map to view crime locations.
    * Click on markers to see detailed information about each crime.

## Machine Learning Model

* The model uses `TfidfVectorizer` to convert text descriptions into numerical features and `LogisticRegression` to predict crime categories.
* Model performance metrics will be added here.
* Limitations: The accuracy of the crime prediction is dependent on the quality of the training data.

