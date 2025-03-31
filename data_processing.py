import re
from datetime import datetime
from typing import Dict, Any, BinaryIO, Union
import pandas as pd
import joblib
from PyPDF2 import PdfReader
import altair as alt

# Constants
COLUMN_LIST = [
    'Dates', 'Category', 'Severity', 'Descript',
    'DayOfWeek', 'PdDistrict', 'Resolution',
    'Address', 'Latitude', 'Longitude'
]

# Severity map
SEVERITY_MAP = {
    "NON-CRIMINAL": 1,
    "SUSPICIOUS OCC": 1,
    "MISSING PERSON": 1,
    "RUNAWAY": 1,
    "RECOVERED VEHICLE": 1,
    "WARRANTS": 2,
    "OTHER OFFENSES": 2,
    "VANDALISM": 2,
    "TRESPASS": 2,
    "DISORDERLY CONDUCT": 2,
    "BAD CHECKS": 2,
    "LARCENY/THEFT": 3,
    "VEHICLE THEFT": 3,
    "FORGERY/COUNTERFEITING": 3,
    "DRUG/NARCOTIC": 3,
    "STOLEN PROPERTY": 3,
    "FRAUD": 3,
    "BRIBERY": 3,
    "EMBEZZLEMENT": 3,
    "ROBBERY": 4,
    "WEAPON LAWS": 4,
    "BURGLARY": 4,
    "EXTORTION": 4,
    "KIDNAPPING": 5,
    "ARSON": 5
}

# Regex patterns for extraction
PATTERN = {
    "Report Number": r"Report Number:\s*([^\n]+)",
    "Date & Time": r"Date & Time:\s*([^\n]+)",
    "Category": r"Category:\s*([^\n]+)",
    "Descript": r"Detailed Description:\s*([\s\S]+?)\nPolice District:",
    "PdDistrict": r"Police District:\s*([^\n]+)",
    "Resolution": r"Resolution:\s*([^\n]+)",
    "Address": r"Incident Location:\s*([^\n]+)",
    "Coordinates": r"Coordinates:\s*\(([^)]+)\)",
}

# This function takes a PDF file-like object and returns a dictionary of extracted fields.
# The "BinaryIO" type hints that the input is a binary file-like object (like one uploaded via Streamlit).
# The "-> Dict[str, Any]" means the function will return a dictionary with string keys and values of any type.
def extract_from_pdf(pdf_file: BinaryIO) -> Dict[str, Any]:
    reader = PdfReader(pdf_file)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

    report_data = {}
    for key, regex in PATTERN.items():
        match = re.search(regex, text)
        report_data[key] = match.group(1).strip() if match else None

    date_string = report_data.get('Date & Time', '')
    if date_string:
        cleaned_date = re.sub(r'\s+', '', date_string)
        try:
            date_obj = datetime.strptime(cleaned_date, '%Y-%m-%d%H:%M:%S')
            report_data['DayOfWeek'] = date_obj.strftime('%A')
        except ValueError:
            report_data['DayOfWeek'] = None

    if 'Coordinates' in report_data and report_data['Coordinates']:
        try:
            lat, lon = report_data['Coordinates'].split(', ')
            report_data['Latitude'] = lat
            report_data['Longitude'] = lon
        except ValueError:
            report_data['Latitude'] = None
            report_data['Longitude'] = None
    else:
        report_data['Latitude'] = None
        report_data['Longitude'] = None

    return {
        'Dates': report_data.get('Date & Time'),
        'Category': report_data.get('Category'),
        'Descript': report_data.get('Descript'),
        'DayOfWeek': report_data.get('DayOfWeek'),
        'PdDistrict': report_data.get('PdDistrict'),
        'Resolution': report_data.get('Resolution'),
        'Address': report_data.get('Address'),
        'Latitude': report_data.get('Latitude'),
        'Longitude': report_data.get('Longitude'),
    }

def load_model(path: str):
    return joblib.load(path)

def predict_category(model, description: str) -> Union[str, None]:
    if description:
        return model.predict([description])[0]
    return None


def standardize_pdf_record(record: Dict[str, Any], model) -> pd.DataFrame:
    # Convert all string fields to uppercase first
    record = {k: v.upper() if isinstance(v, str) else v for k, v in record.items()}
    
    # Then apply classification and severity
    record['Category'] = predict_category(model, record.get('Descript'))
    record['Severity'] = SEVERITY_MAP.get(record['Category'], 0)
    record = {k: v.upper() if isinstance(v, str) else v for k, v in record.items()}
    return pd.DataFrame([record])[COLUMN_LIST]

def process_comp_data(df):
    df.rename(columns={
            'Latitude (Y)': 'Longitude',
            'Longitude (X)': 'Latitude'
        }, inplace=True)
    df['Severity'] = df['Category'].map(SEVERITY_MAP).fillna(0).astype(int)
    df = df[COLUMN_LIST]
    df = df.applymap(lambda x: x.upper() if isinstance(x, str) else x)
    df['Dates'] = pd.to_datetime(df['Dates'], errors='coerce')
    return df