import os
import ee
import numpy as np
import datetime
import joblib
import logging
import pandas as pd
from typing import List
from fastapi import FastAPI, HTTPException, File, UploadFile, Depends
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)

# Inisialisasi kredensial Google Earth Engine
google_credentials = "/etc/secrets/ee-ulikhasanah16-743b3ec3e985.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_credentials

try:
    service_account = "ee-ulikhasanah16-743b3ec3e985@developer.gserviceaccount.com"
    credentials = ee.ServiceAccountCredentials(service_account, google_credentials)
    ee.Initialize(credentials)
    logging.info("Google Earth Engine initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize Earth Engine: {e}")
    raise

try:
    catboost_model = joblib.load("catboost_chlor_a.pkl")
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    raise

scaler = joblib.load("scaler.pkl")

SATELLITES = {"Sentinel-2": "COPERNICUS/S2_SR"}
BANDS = {"Sentinel-2": {"Red": "B4", "NIR": "B8", "SWIR1": "B11", "SWIR2": "B12", "Blue": "B2", "Green": "B3"}}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Location(BaseModel):
    lat: float
    lon: float
    date: str = None

class MultiLocation(BaseModel):
    locations: List[Location]

def process_request(lat, lon, date):
    # Fungsi yang mengambil data dari GEE dan memprosesnya seperti yang telah Anda buat sebelumnya
    pass  # Implementasi tetap sama seperti sebelumnya

@app.post("/predict")
def predict_chlorophyll(data: Location):
    return process_request(data.lat, data.lon, data.date)

@app.post("/predict-multi")
def predict_multiple_chlorophyll(data: MultiLocation):
    results = [process_request(loc.lat, loc.lon, loc.date) for loc in data.locations]
    return results

@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file) if file.filename.endswith('.csv') else pd.read_excel(file.file)
        if not all(col in df.columns for col in ['lat', 'lon', 'date']):
            raise HTTPException(status_code=400, detail="File must contain 'lat', 'lon', and 'date' columns.")
        
        results = [process_request(row['lat'], row['lon'], row['date']) for _, row in df.iterrows()]
        return results
    except Exception as e:
        logging.error(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail="Error processing file.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
